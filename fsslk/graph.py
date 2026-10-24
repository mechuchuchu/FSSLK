"""Capture-only graph extraction for PyTorch modules wrapped by FSSLK.

``fsslk.wrap(module, example_inputs)`` exports the wrapped part of a model with
``torch.export`` and turns the result into a plain, JSON-serializable
description: every operation, the tensors on its edges, an estimate of its
FLOPs, and for every possible two-stage split the activation bytes that would
have to cross the network.

This is deliberately *capture only*. The wrapper delegates ``forward`` to the
original module, so wrapping never changes what the model computes. Nothing is
shipped to workers yet, and the serialized form is data (op names, shapes,
dtypes), never pickle, so it can later be validated before a worker runs it.

Known limits of this first version:

* forward graph only (no autograd graph);
* control flow and higher-order ops are rejected with a ``CaptureError``;
* FLOPs follow ``torch.utils.flop_counter`` conventions: matmul-like ops
  (linear, mm, addmm, bmm, matmul), convolutions and scaled dot product
  attention are counted, pointwise ops count as zero. The capture cross-checks
  the total against ``FlopCounterMode`` so unmodeled heavy ops show up;
* wrapping nests the module, so ``state_dict`` keys gain a ``module.`` prefix.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn

from .protocol import MAX_PAYLOAD_BYTES
from .scheduler import NodeCapacity
from .tensor import _DTYPE_NAMES

GRAPH_FORMAT_VERSION = 1

_NODE_KINDS = frozenset({"input", "param", "buffer", "other", "compute", "output"})
_KIND_BY_INPUT_SPEC = {
    "USER_INPUT": "input",
    "PARAMETER": "param",
    "BUFFER": "buffer",
    "CONSTANT_TENSOR": "buffer",
}
_WIRE_DTYPES = frozenset(_DTYPE_NAMES.values())

# Position of the argument whose last dimension is the contraction size.
_MATMUL_CONTRACTION_ARG = {
    "aten.linear.default": 0,
    "aten.mm.default": 0,
    "aten.bmm.default": 0,
    "aten.matmul.default": 0,
    "aten.addmm.default": 1,
    "aten.baddbmm.default": 1,
}
_CONV_OPS = frozenset(
    {
        "aten.conv1d.default",
        "aten.conv2d.default",
        "aten.conv3d.default",
        "aten.convolution.default",
    }
)
_SDPA_OP = "aten.scaled_dot_product_attention.default"


class CaptureError(RuntimeError):
    """The wrapped module could not be captured as a supported graph.

    ``diagnostics`` holds whatever ``torch.export`` printed to stderr while
    failing (it dumps the whole failing graph), so it does not flood the
    console but is still available for debugging.
    """

    def __init__(self, message: str, diagnostics: str = "") -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class TensorInfo:
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass(frozen=True)
class GraphNode:
    name: str
    kind: str  # input | param | buffer | other | compute | output
    op: str
    inputs: tuple[str, ...]
    outputs: tuple[TensorInfo, ...]
    flops: int = 0

    @property
    def out_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.outputs)


@dataclass(frozen=True)
class LinkProfile:
    """One network link, costed with the same model as ``allocate_chunks``.

    A transfer pays one RTT plus ``nbytes`` at the bandwidth reduced linearly
    by packet loss (never below 1% of nominal).
    """

    name: str
    bandwidth_mbps: float
    rtt_ms: float = 0.0
    packet_loss_pct: float = 0.0

    def transfer_seconds(self, nbytes: int) -> float:
        loss = min(max(self.packet_loss_pct, 0.0), 100.0)
        loss_factor = max(0.01, 1.0 - loss / 100.0)
        rate = self.bandwidth_mbps * 1_000_000 / 8 * loss_factor
        return max(self.rtt_ms, 0.0) / 1000 + nbytes / rate

    @classmethod
    def from_capacity(cls, capacity: NodeCapacity) -> "LinkProfile":
        if capacity.bandwidth_mbps is None or capacity.bandwidth_mbps <= 0:
            raise ValueError(f"node {capacity.node_id} has no measured bandwidth")
        return cls(
            name=capacity.node_id,
            bandwidth_mbps=capacity.bandwidth_mbps,
            rtt_ms=capacity.rtt_ms or 0.0,
            packet_loss_pct=capacity.packet_loss_pct or 0.0,
        )


# Measured through the docker-compose netem profiles (daily_report/2026-10-24.md).
EXAMPLE_LINKS = (
    LinkProfile("fast", bandwidth_mbps=86.22, rtt_ms=5.19, packet_loss_pct=0.0),
    LinkProfile("medium", bandwidth_mbps=8.03, rtt_ms=56.27, packet_loss_pct=1.0),
    LinkProfile("slow", bandwidth_mbps=0.42, rtt_ms=332.2, packet_loss_pct=6.5),
)


@dataclass(frozen=True)
class CutPoint:
    """A two-stage split: stage A runs up to and including ``after``."""

    after: str
    position: int
    transfer_bytes: int  # activation bytes crossing from stage A to stage B
    crossing: tuple[str, ...]
    flops_before: int
    flops_after: int
    param_bytes_before: int
    param_bytes_after: int

    @property
    def flops_share_before(self) -> float:
        total = self.flops_before + self.flops_after
        return self.flops_before / total if total else 0.0


@dataclass(frozen=True)
class CapturedGraph:
    module_name: str
    nodes: tuple[GraphNode, ...]
    reference_flops: int | None = None

    # ------------------------------------------------------------------ totals
    @property
    def compute_nodes(self) -> tuple[GraphNode, ...]:
        return tuple(node for node in self.nodes if node.kind == "compute")

    @property
    def total_flops(self) -> int:
        return sum(node.flops for node in self.nodes)

    def bytes_of(self, kind: str) -> int:
        return sum(node.out_bytes for node in self.nodes if node.kind == kind)

    @property
    def output_bytes(self) -> int:
        by_name = {node.name: node for node in self.nodes}
        returned = {
            name
            for node in self.nodes
            if node.kind == "output"
            for name in node.inputs
        }
        return sum(by_name[name].out_bytes for name in returned)

    # ------------------------------------------------------------ cut analysis
    def cut_points(self, min_share: float = 0.0) -> list[CutPoint]:
        """Every two-stage split, optionally keeping balanced ones only.

        ``min_share`` drops splits where either side holds less than that
        fraction of the work (FLOPs, or node count when the graph has no
        counted FLOPs). Weights are not streamed per step, so they are
        reported as per-stage bytes instead of crossing bytes.
        """
        if not 0.0 <= min_share <= 0.5:
            raise ValueError("min_share must be between 0 and 0.5")
        compute = self.compute_nodes
        by_name = {node.name: node for node in self.nodes}
        position = {node.name: index for index, node in enumerate(compute)}
        first_use: dict[str, int] = {}
        last_use: dict[str, int] = {}
        for index, node in enumerate(compute):
            for source in node.inputs:
                first_use.setdefault(source, index)
                last_use[source] = index

        total_flops = sum(node.flops for node in compute)
        weights = [node.flops if total_flops else 1 for node in compute]
        total_weight = sum(weights)

        cuts: list[CutPoint] = []
        for index in range(len(compute) - 1):
            share = sum(weights[: index + 1]) / total_weight if total_weight else 0.0
            if share < min_share or share > 1.0 - min_share:
                continue
            crossing: list[str] = []
            params_before = params_after = 0
            for source, last in last_use.items():
                producer = by_name[source]
                if producer.kind in ("param", "buffer"):
                    if first_use[source] <= index:
                        params_before += producer.out_bytes
                    if last > index:
                        params_after += producer.out_bytes
                elif last > index and (
                    producer.kind == "input"
                    or (producer.kind == "compute" and position[source] <= index)
                ):
                    crossing.append(source)
            cuts.append(
                CutPoint(
                    after=compute[index].name,
                    position=index,
                    transfer_bytes=sum(by_name[name].out_bytes for name in crossing),
                    crossing=tuple(crossing),
                    flops_before=sum(node.flops for node in compute[: index + 1]),
                    flops_after=sum(node.flops for node in compute[index + 1 :]),
                    param_bytes_before=params_before,
                    param_bytes_after=params_after,
                )
            )
        return cuts

    # ------------------------------------------------- wire protocol checks
    def unsupported_dtypes(self) -> list[str]:
        """Dtypes the current wire protocol cannot encode (only int64/float32)."""
        return sorted(
            {
                tensor.dtype
                for node in self.nodes
                for tensor in node.outputs
                if tensor.dtype not in _WIRE_DTYPES
            }
        )

    def oversized_tensors(self) -> list[tuple[str, int]]:
        """Single tensors larger than one protocol frame (64 MiB)."""
        return [
            (node.name, tensor.nbytes)
            for node in self.nodes
            for tensor in node.outputs
            if tensor.nbytes > MAX_PAYLOAD_BYTES
        ]

    # ------------------------------------------------------- serialization
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": GRAPH_FORMAT_VERSION,
            "module": self.module_name,
            "reference_flops": self.reference_flops,
            "nodes": [
                {
                    "name": node.name,
                    "kind": node.kind,
                    "op": node.op,
                    "inputs": list(node.inputs),
                    "outputs": [
                        {
                            "shape": list(tensor.shape),
                            "dtype": tensor.dtype,
                            "nbytes": tensor.nbytes,
                        }
                        for tensor in node.outputs
                    ],
                    "flops": node.flops,
                }
                for node in self.nodes
            ],
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "CapturedGraph":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, data: Any) -> "CapturedGraph":
        """Rebuild a graph, validating it as untrusted data."""
        if not isinstance(data, dict) or data.get("version") != GRAPH_FORMAT_VERSION:
            raise ValueError("unsupported graph format")
        module_name = data.get("module")
        if not isinstance(module_name, str):
            raise ValueError("graph module name must be a string")
        reference = data.get("reference_flops")
        if reference is not None and not _is_count(reference):
            raise ValueError("reference_flops must be a non-negative integer or null")
        raw_nodes = data.get("nodes")
        if not isinstance(raw_nodes, list):
            raise ValueError("graph nodes must be a list")

        nodes: list[GraphNode] = []
        seen: set[str] = set()
        for raw in raw_nodes:
            if not isinstance(raw, dict):
                raise ValueError("graph node must be an object")
            name, kind, op = raw.get("name"), raw.get("kind"), raw.get("op")
            if not isinstance(name, str) or not name or name in seen:
                raise ValueError(f"invalid or duplicate node name: {name!r}")
            if kind not in _NODE_KINDS:
                raise ValueError(f"unknown node kind: {kind!r}")
            if not isinstance(op, str):
                raise ValueError(f"node {name} has an invalid op")
            inputs = raw.get("inputs")
            if not isinstance(inputs, list) or any(
                not isinstance(source, str) or source not in seen for source in inputs
            ):
                raise ValueError(f"node {name} must only read earlier nodes")
            flops = raw.get("flops", 0)
            if not _is_count(flops):
                raise ValueError(f"node {name} has invalid flops")
            raw_outputs = raw.get("outputs")
            if not isinstance(raw_outputs, list):
                raise ValueError(f"node {name} outputs must be a list")
            outputs = []
            for raw_tensor in raw_outputs:
                shape = raw_tensor.get("shape") if isinstance(raw_tensor, dict) else None
                dtype = raw_tensor.get("dtype") if isinstance(raw_tensor, dict) else None
                nbytes = raw_tensor.get("nbytes") if isinstance(raw_tensor, dict) else None
                if (
                    not isinstance(shape, list)
                    or len(shape) > 32
                    or not all(_is_count(dim) for dim in shape)
                    or not isinstance(dtype, str)
                    or not _is_count(nbytes)
                ):
                    raise ValueError(f"node {name} has an invalid tensor description")
                outputs.append(TensorInfo(tuple(shape), dtype, nbytes))
            seen.add(name)
            nodes.append(
                GraphNode(name, kind, op, tuple(inputs), tuple(outputs), flops)
            )
        return cls(module_name, tuple(nodes), reference)

    # --------------------------------------------------------------- report
    def report(
        self,
        links: Sequence[LinkProfile] = EXAMPLE_LINKS,
        *,
        top_nodes: int = 5,
        top_cuts: int = 5,
        min_share: float = 0.2,
    ) -> str:
        compute = self.compute_nodes
        params = [node for node in self.nodes if node.kind == "param"]
        buffers = [node for node in self.nodes if node.kind == "buffer"]
        inputs = [node for node in self.nodes if node.kind == "input"]

        lines = [f"FSSLK graph: {self.module_name} (torch.export)"]
        lines.append(
            f"compute nodes {len(compute)} | inputs {len(inputs)} "
            f"({_fmt_bytes(self.bytes_of('input'))}) | params {len(params)} "
            f"({_fmt_bytes(self.bytes_of('param'))}) | buffers {len(buffers)}"
        )
        lines.append(f"FLOPs {_fmt_flops(self.total_flops)} {self._reference_note()}")
        largest = max((node.out_bytes for node in compute), default=0)
        lines.append(
            f"largest activation {_fmt_bytes(largest)} | "
            f"result {_fmt_bytes(self.output_bytes)}"
        )

        heaviest = sorted(compute, key=lambda node: (node.flops, node.out_bytes), reverse=True)
        lines += ["", "Heaviest nodes"]
        lines += _table(
            ["node", "op", "FLOPs", "output"],
            [
                [node.name, node.op, _fmt_flops(node.flops), _fmt_bytes(node.out_bytes)]
                for node in heaviest[:top_nodes]
            ],
        )

        cuts = sorted(
            self.cut_points(min_share),
            key=lambda cut: (cut.transfer_bytes, abs(cut.flops_share_before - 0.5)),
        )
        lines += [
            "",
            f"Cheapest 2-stage cuts (each side keeps >= {min_share:.0%} of the work)",
        ]
        if cuts:
            lines += _table(
                ["after", "crossing", "A/B", *[link.name for link in links]],
                [
                    [
                        cut.after,
                        f"{_fmt_bytes(cut.transfer_bytes)} x{len(cut.crossing)}",
                        f"{cut.flops_share_before:.0%}/{1 - cut.flops_share_before:.0%}",
                        *[
                            _fmt_seconds(link.transfer_seconds(cut.transfer_bytes))
                            for link in links
                        ],
                    ]
                    for cut in cuts[:top_cuts]
                ],
            )
        else:
            lines.append("  (no balanced cut available)")

        lines += ["", self._wire_note()]
        return "\n".join(lines)

    def _reference_note(self) -> str:
        if self.reference_flops is None:
            return "(no FlopCounterMode cross-check)"
        if self.reference_flops == self.total_flops:
            return "(matches FlopCounterMode)"
        return (
            f"(FlopCounterMode counts {_fmt_flops(self.reference_flops)}: "
            "the two disagree on some ops)"
        )

    def _wire_note(self) -> str:
        problems = []
        if self.unsupported_dtypes():
            problems.append(f"unsupported dtypes {', '.join(self.unsupported_dtypes())}")
        if self.oversized_tensors():
            problems.append(f"{len(self.oversized_tensors())} tensors over the 64 MiB frame limit")
        if problems:
            return "Wire protocol: NOT ready - " + "; ".join(problems)
        return "Wire protocol: ok (int64/float32 only, every tensor fits one 64 MiB frame)"


def capture_graph(
    module: nn.Module,
    args: Sequence[Any] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    reference: bool = True,
) -> CapturedGraph:
    """Export ``module`` for the given example inputs and describe the graph."""
    kwargs = kwargs or {}
    printed = io.StringIO()
    try:
        with contextlib.redirect_stderr(printed):
            program = torch.export.export(module, tuple(args), kwargs, strict=False)
    except Exception as exc:  # torch.export raises many unrelated error types
        raise CaptureError(
            f"torch.export could not capture {type(module).__name__}: {exc}",
            diagnostics=printed.getvalue(),
        ) from exc
    if printed.getvalue():  # e.g. warnings from a successful export: do not swallow them
        sys.stderr.write(printed.getvalue())

    kinds: dict[str, str] = {}
    for spec in program.graph_signature.input_specs:
        arg_name = getattr(spec.arg, "name", None)
        if arg_name is not None:
            kinds[arg_name] = _KIND_BY_INPUT_SPEC.get(spec.kind.name, "other")

    nodes: list[GraphNode] = []
    for fx_node in program.graph.nodes:
        if fx_node.op == "placeholder":
            kind, op = kinds.get(fx_node.name, "input"), "placeholder"
        elif fx_node.op == "output":
            kind, op = "output", "output"
        elif fx_node.op == "call_function":
            kind, op = "compute", _op_name(fx_node.target)
            if op.startswith("higher_order"):
                raise CaptureError(f"control flow is not supported yet ({op})")
        else:
            raise CaptureError(f"unsupported graph node type: {fx_node.op}")
        nodes.append(
            GraphNode(
                name=fx_node.name,
                kind=kind,
                op=op,
                inputs=tuple(dict.fromkeys(src.name for src in fx_node.all_input_nodes)),
                outputs=_tensor_infos(fx_node.meta.get("val")),
                flops=_estimate_flops(fx_node, op) if kind == "compute" else 0,
            )
        )

    reference_flops = _reference_flops(module, args, kwargs) if reference else None
    return CapturedGraph(type(module).__name__, tuple(nodes), reference_flops)


class WrappedModule(nn.Module):
    """Runs the wrapped module unchanged and keeps its captured graph."""

    def __init__(
        self,
        module: nn.Module,
        example_inputs: Any = None,
        example_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.module = module
        self._graph: CapturedGraph | None = None
        self.capture_error: CaptureError | None = None
        if example_inputs is not None:
            self.capture(example_inputs, example_kwargs)

    def capture(
        self, example_inputs: Any, example_kwargs: dict[str, Any] | None = None
    ) -> CapturedGraph:
        args = (example_inputs,) if isinstance(example_inputs, torch.Tensor) else tuple(example_inputs)
        self._graph = capture_graph(self.module, args, example_kwargs)
        self.capture_error = None
        return self._graph

    @property
    def graph(self) -> CapturedGraph:
        if self._graph is None:
            raise RuntimeError(
                "no graph captured yet: pass example_inputs to wrap(), call "
                ".capture(example_inputs) or run forward once"
            )
        return self._graph

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._graph is None and self.capture_error is None:
            try:
                self.capture(args, kwargs)
            except CaptureError as exc:
                # Capture-only mode must never break the model itself.
                self.capture_error = exc
                warnings.warn(f"FSSLK graph capture skipped: {exc}", stacklevel=2)
        return self.module(*args, **kwargs)


def wrap(
    module: nn.Module,
    example_inputs: Any = None,
    example_kwargs: dict[str, Any] | None = None,
) -> WrappedModule:
    """Wrap ``module`` so FSSLK can capture the graph of that part of a model.

    With ``example_inputs`` the graph is captured immediately (errors raise
    ``CaptureError``); without them it is captured on the first forward call
    (errors become a warning and the forward pass still runs).
    """
    return WrappedModule(module, example_inputs, example_kwargs)


# ---------------------------------------------------------------------- helpers
def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _op_name(target: Any) -> str:
    if isinstance(target, torch._ops.OpOverload):
        return str(target)
    return getattr(target, "__name__", None) or repr(target)


def _tensor_infos(value: Any) -> tuple[TensorInfo, ...]:
    if isinstance(value, torch.Tensor):
        shape = tuple(int(dim) for dim in value.shape)
        dtype = str(value.dtype).removeprefix("torch.")
        return (TensorInfo(shape, dtype, math.prod(shape) * value.element_size()),)
    if isinstance(value, (tuple, list)):
        return tuple(info for item in value for info in _tensor_infos(item))
    return ()


def _arg_shape(node: torch.fx.Node, index: int) -> tuple[int, ...] | None:
    if index >= len(node.args) or not isinstance(node.args[index], torch.fx.Node):
        return None
    value = node.args[index].meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    return tuple(int(dim) for dim in value.shape)


def _estimate_flops(node: torch.fx.Node, op: str) -> int:
    out = node.meta.get("val")
    if not isinstance(out, torch.Tensor):
        return 0
    out_numel = math.prod(int(dim) for dim in out.shape)

    contraction_arg = _MATMUL_CONTRACTION_ARG.get(op)
    if contraction_arg is not None:
        shape = _arg_shape(node, contraction_arg)
        return 2 * out_numel * shape[-1] if shape else 0
    if op in _CONV_OPS:
        weight = _arg_shape(node, 1)
        transposed = op == "aten.convolution.default" and len(node.args) > 6 and node.args[6]
        if weight is None or len(weight) < 3 or transposed:
            return 0
        return 2 * out_numel * math.prod(weight[1:])
    if op == _SDPA_OP:
        query, key, value = (_arg_shape(node, index) for index in range(3))
        if query is None or key is None or value is None or len(query) < 2 or len(key) < 2:
            return 0
        return 2 * math.prod(query[:-2]) * query[-2] * key[-2] * (query[-1] + value[-1])
    return 0


def _reference_flops(module: nn.Module, args: Sequence[Any], kwargs: dict[str, Any]) -> int | None:
    """FlopCounterMode total from one real forward pass.

    Skipped in training mode: running the module there would update batch norm
    statistics and consume dropout randomness, i.e. wrapping would change the
    model. In eval mode the RNG is restored around the run anyway.
    """
    if module.training:
        return None
    from torch.utils.flop_counter import FlopCounterMode

    try:
        with torch.no_grad(), torch.random.fork_rng(devices=[]), FlopCounterMode(
            display=False, custom_mapping=_cpu_attention_mapping()
        ) as counter:
            module(*args, **kwargs)
        return int(counter.get_total_flops())
    except Exception:
        return None


def _cpu_attention_mapping() -> dict[Any, Any]:
    """FlopCounterMode has no formula for the CPU attention kernel (it counts 0).

    Register torch's own ``sdpa_flop_count`` for it so the cross-check does not
    miss attention. If this torch version lacks either piece, the cross-check
    simply reports the disagreement instead.
    """
    try:
        from torch.utils.flop_counter import sdpa_flop_count

        op = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu
    except (ImportError, AttributeError):
        return {}

    def formula(query_shape, key_shape, value_shape, *args, **kwargs):
        return sdpa_flop_count(query_shape, key_shape, value_shape)

    return {op: formula}


def _fmt_bytes(count: float) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB"):
        if value < 1024:
            return f"{value:.0f} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def _fmt_flops(count: float) -> str:
    value = float(count)
    for unit in ("FLOP", "KFLOP", "MFLOP", "GFLOP"):
        if value < 1000:
            return f"{value:.0f} {unit}" if unit == "FLOP" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.2f} TFLOP"


def _fmt_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 120:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[col]) for row in [header, *rows]) for col in range(len(header))]
    return [
        "  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        for row in [header, *rows]
    ]
