"""Cut a captured graph into stages and run a stage on a worker.

``split_two_stage`` turns a ``CapturedGraph`` and a cut point into two
``StageSpec`` objects. A spec is plain data: the tensors it needs, the weights
it holds, its operations (names from ``ops.ALLOWED_OPS`` with encoded
arguments) and the tensors it returns. ``StageSpec.from_dict`` treats its input
as untrusted, because workers build specs from what the network sends them.

``StageRuntime`` executes a spec: weights are loaded once with ``put_params``
and then ``run`` takes activations in and gives activations out, so only
activations cross the network on each step.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .graph import CapturedGraph, CutPoint, GraphNode
from .ops import ALLOWED_OPS, call_refs, execute_op
from .protocol import MAX_PAYLOAD_BYTES
from .tensor import _DTYPE_NAMES

STAGE_FORMAT_VERSION = 1
MAX_STAGE_NODES = 50_000
MAX_STAGE_TENSORS = 8192
MAX_STAGE_PARAM_BYTES = 4 * 1024**3
MAX_TENSOR_DIMS = 8

_STAGE_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_WIRE_DTYPES = {name: dtype for dtype, name in _DTYPE_NAMES.items()}  # name -> torch dtype


class StageError(ValueError):
    """The graph cannot be cut as requested, or a stage spec is invalid."""


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * torch.empty((), dtype=_WIRE_DTYPES[self.dtype]).element_size()


@dataclass(frozen=True)
class StageNode:
    name: str
    op: str
    call: dict[str, Any]


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    inputs: tuple[TensorSpec, ...]
    params: tuple[TensorSpec, ...]
    nodes: tuple[StageNode, ...]
    outputs: tuple[str, ...]

    # ---------------------------------------------------------- serialization
    def to_dict(self) -> dict[str, Any]:
        def tensor(spec: TensorSpec) -> dict[str, Any]:
            return {"name": spec.name, "shape": list(spec.shape), "dtype": spec.dtype}

        return {
            "version": STAGE_FORMAT_VERSION,
            "stage_id": self.stage_id,
            "inputs": [tensor(spec) for spec in self.inputs],
            "params": [tensor(spec) for spec in self.params],
            "nodes": [
                {"name": node.name, "op": node.op, "call": node.call} for node in self.nodes
            ],
            "outputs": list(self.outputs),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "StageSpec":
        """Rebuild and fully validate a spec received from the network."""
        if not isinstance(data, dict) or data.get("version") != STAGE_FORMAT_VERSION:
            raise StageError("unsupported stage format")
        stage_id = data.get("stage_id")
        if not isinstance(stage_id, str) or not _STAGE_ID.fullmatch(stage_id):
            raise StageError("invalid stage id")

        defined: set[str] = set()

        def read_tensors(key: str) -> tuple[TensorSpec, ...]:
            raw = data.get(key)
            if not isinstance(raw, list) or len(raw) > MAX_STAGE_TENSORS:
                raise StageError(f"stage {key} must be a list of at most {MAX_STAGE_TENSORS}")
            specs = []
            for entry in raw:
                if not isinstance(entry, dict):
                    raise StageError(f"stage {key} entry must be an object")
                name, shape, dtype = entry.get("name"), entry.get("shape"), entry.get("dtype")
                if not _is_name(name) or name in defined:
                    raise StageError(f"invalid or duplicate tensor name: {name!r}")
                if dtype not in _WIRE_DTYPES:
                    raise StageError(f"tensor {name} has unsupported dtype {dtype!r}")
                if (
                    not isinstance(shape, list)
                    or len(shape) > MAX_TENSOR_DIMS
                    or not all(isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape)
                ):
                    raise StageError(f"tensor {name} has an invalid shape")
                spec = TensorSpec(name, tuple(shape), dtype)
                if spec.nbytes > MAX_PAYLOAD_BYTES:
                    raise StageError(f"tensor {name} does not fit one protocol frame")
                defined.add(name)
                specs.append(spec)
            return tuple(specs)

        inputs = read_tensors("inputs")
        params = read_tensors("params")
        if sum(spec.nbytes for spec in params) > MAX_STAGE_PARAM_BYTES:
            raise StageError("stage weights are too large for one worker")

        raw_nodes = data.get("nodes")
        if not isinstance(raw_nodes, list) or len(raw_nodes) > MAX_STAGE_NODES:
            raise StageError(f"stage nodes must be a list of at most {MAX_STAGE_NODES}")
        nodes = []
        for entry in raw_nodes:
            if not isinstance(entry, dict):
                raise StageError("stage node must be an object")
            name, op, call = entry.get("name"), entry.get("op"), entry.get("call")
            if not _is_name(name) or name in defined:
                raise StageError(f"invalid or duplicate node name: {name!r}")
            if op not in ALLOWED_OPS:
                raise StageError(f"operation not allowed: {op!r}")
            try:
                refs = call_refs(call)
            except ValueError as exc:
                raise StageError(f"node {name}: {exc}") from exc
            if any(ref not in defined for ref in refs):
                raise StageError(f"node {name} reads a tensor that is not defined before it")
            defined.add(name)
            nodes.append(StageNode(name, op, call))

        outputs = data.get("outputs")
        if (
            not isinstance(outputs, list)
            or not 1 <= len(outputs) <= 256
            or any(name not in defined for name in outputs)
        ):
            raise StageError("stage outputs must name defined tensors")
        return cls(stage_id, inputs, params, tuple(nodes), tuple(outputs))


def _is_name(value: Any) -> bool:
    return isinstance(value, str) and value.isidentifier() and len(value) <= 128


# --------------------------------------------------------------------- cutting
def shippable_cuts(graph: CapturedGraph, min_share: float = 0.2) -> list[CutPoint]:
    """Cut points where every crossing value is one plain tensor."""
    by_name = {node.name: node for node in graph.nodes}
    return [
        cut
        for cut in graph.cut_points(min_share)
        if all(len(by_name[name].outputs) == 1 for name in cut.crossing)
    ]


def choose_cut(graph: CapturedGraph, min_share: float = 0.2) -> CutPoint:
    """The cut that moves the fewest bytes, preferring a balanced split on ties."""
    cuts = shippable_cuts(graph, min_share)
    if not cuts:
        raise StageError("no balanced cut where every crossing value is a single tensor")
    return min(cuts, key=lambda cut: (cut.transfer_bytes, abs(cut.flops_share_before - 0.5)))


def split_two_stage(
    graph: CapturedGraph, cut_after: str, stage_prefix: str = "stage"
) -> tuple[StageSpec, StageSpec]:
    """Stage A runs up to and including ``cut_after``; stage B runs the rest."""
    problems = graph.shippable_problems()
    if problems:
        raise StageError("graph cannot be shipped to workers: " + "; ".join(problems))
    compute = graph.compute_nodes
    names = [node.name for node in compute]
    if cut_after not in names[:-1]:
        raise StageError(f"cannot cut after {cut_after!r}: pick a compute node that is not the last")
    first_b = names.index(cut_after) + 1
    by_name = {node.name: node for node in graph.nodes}
    produced_in_a = set(names[:first_b])
    returned = graph.output_names()
    if any(name in produced_in_a for name in returned):
        raise StageError("the graph returns a tensor computed before the cut")

    nodes_a, nodes_b = compute[:first_b], compute[first_b:]
    reads_a, reads_b = _reads(nodes_a), _reads(nodes_b)

    inputs_a = [name for name in reads_a if by_name[name].kind == "input"]
    inputs_b = [
        name for name in reads_b if name in produced_in_a or by_name[name].kind == "input"
    ]
    crossing = [name for name in inputs_b if name in produced_in_a]
    for name in crossing:
        if len(by_name[name].outputs) != 1:
            raise StageError(f"the cut would send {name}, which is not a single tensor")

    spec_a = StageSpec(
        f"{stage_prefix}-a",
        tuple(_tensor_spec(by_name[name]) for name in inputs_a),
        tuple(_tensor_spec(by_name[name]) for name in reads_a if by_name[name].kind in ("param", "buffer")),
        tuple(StageNode(node.name, node.op, node.call) for node in nodes_a),
        tuple(crossing),
    )
    spec_b = StageSpec(
        f"{stage_prefix}-b",
        tuple(_tensor_spec(by_name[name]) for name in inputs_b),
        tuple(_tensor_spec(by_name[name]) for name in reads_b if by_name[name].kind in ("param", "buffer")),
        tuple(StageNode(node.name, node.op, node.call) for node in nodes_b),
        tuple(dict.fromkeys(returned)),
    )
    return spec_a, spec_b


def stage_flops(graph: CapturedGraph, spec: StageSpec) -> int:
    in_stage = {node.name for node in spec.nodes}
    return sum(node.flops for node in graph.nodes if node.name in in_stage)


def _reads(nodes: Sequence[GraphNode]) -> list[str]:
    """Distinct names read by ``nodes`` that they do not compute themselves."""
    own = {node.name for node in nodes}
    ordered = dict.fromkeys(ref for node in nodes for ref in call_refs(node.call))
    return [name for name in ordered if name not in own]


def _tensor_spec(node: GraphNode) -> TensorSpec:
    if len(node.outputs) != 1:
        raise StageError(f"{node.name} is not a single tensor")
    info = node.outputs[0]
    if info.dtype not in _WIRE_DTYPES:
        raise StageError(
            f"{node.name} has dtype {info.dtype}; the wire protocol carries only "
            + "/".join(sorted(_WIRE_DTYPES))
        )
    return TensorSpec(node.name, info.shape, info.dtype)


# ------------------------------------------------------------------- execution
class StageRuntime:
    """A validated stage with its weights, ready to run."""

    def __init__(self, spec: StageSpec) -> None:
        self.spec = spec
        self.params: dict[str, torch.Tensor] = {}
        self._param_specs = {item.name: item for item in spec.params}
        self._input_specs = {item.name: item for item in spec.inputs}

    @property
    def missing_params(self) -> list[str]:
        return [name for name in self._param_specs if name not in self.params]

    def put_params(self, tensors: dict[str, torch.Tensor]) -> None:
        for name, tensor in tensors.items():
            self._check(self._param_specs.get(name), name, tensor, "weight")
            self.params[name] = tensor

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.missing_params:
            raise StageError(f"weights not loaded yet: {len(self.missing_params)} missing")
        if set(inputs) != set(self._input_specs):
            raise StageError(
                f"expected inputs {sorted(self._input_specs)}, got {sorted(inputs)}"
            )
        for name, tensor in inputs.items():
            self._check(self._input_specs[name], name, tensor, "input")

        env: dict[str, Any] = {**self.params, **inputs}
        with torch.no_grad():
            for node in self.spec.nodes:
                env[node.name] = execute_op(node.op, node.call, env)
        results = {name: env[name] for name in self.spec.outputs}
        for name, value in results.items():
            if not isinstance(value, torch.Tensor):
                raise StageError(f"stage output {name} is not a tensor")
        return results

    @staticmethod
    def _check(spec: TensorSpec | None, name: str, tensor: torch.Tensor, what: str) -> None:
        if spec is None:
            raise StageError(f"unknown {what}: {name!r}")
        if tuple(tensor.shape) != spec.shape or tensor.dtype != _WIRE_DTYPES[spec.dtype]:
            raise StageError(
                f"{what} {name} is {tuple(tensor.shape)} {tensor.dtype}, "
                f"expected {spec.shape} {spec.dtype}"
            )
