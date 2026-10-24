"""Run a captured model across two workers as a two-stage pipeline.

The driver captures the model's graph, cuts it at the cheapest balanced point,
ships each stage (its operations and its weights, once) to a worker through the
coordinator, then streams micro-batches through: stage A on one worker, stage B
on the other, activations travelling worker -> coordinator -> worker.

Workers connect *to* the coordinator, so the cut tensor crosses two links (the
first worker's uplink and the second worker's downlink), not one.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import statistics
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from .bundle import decode_bundle, encode_bundle, split_bundles
from .graph import CapturedGraph, LinkProfile, capture_graph
from .protocol import receive_frame, send_frame
from .stage import StageSpec, choose_cut, split_two_stage, stage_flops


class PipelineError(RuntimeError):
    """The pipeline could not be set up or a worker call failed."""


# ------------------------------------------------------------------ coordinator
def request(
    host: str,
    port: int,
    message: dict[str, Any],
    payload: bytes = b"",
    timeout: float = 900,
) -> tuple[dict[str, Any], bytes]:
    """One client request to the coordinator; the reply may carry a payload."""
    with socket.create_connection((host, port), timeout=10) as conn:
        conn.settimeout(timeout)
        stream = conn.makefile("rb")
        try:
            send_frame(conn, {"role": "client", **message}, payload)
            frame = receive_frame(stream)
        finally:
            stream.close()
    if frame is None:
        raise ConnectionError("coordinator closed the connection without a response")
    return frame


def call_worker(
    host: str,
    port: int,
    worker: str,
    command: str,
    fields: dict[str, Any] | None = None,
    payload: bytes = b"",
) -> tuple[dict[str, Any], bytes]:
    response, reply_payload = request(
        host, port, {"command": command, "worker": worker, **(fields or {})}, payload
    )
    if not response.get("ok"):
        raise PipelineError(f"{worker}: {command} failed: {response.get('error', 'unknown error')}")
    return response, reply_payload


def wait_for_workers(
    host: str, port: int, workers: list[str], timeout: float = 90
) -> dict[str, dict[str, Any]]:
    """Wait until every worker is registered and the coordinator has measured its link."""
    deadline = time.monotonic() + timeout
    while True:
        response, _ = request(host, port, {"command": "status"}, timeout=15)
        nodes = {node["node_id"]: node for node in response.get("nodes", [])}
        ready = all(
            name in nodes
            and nodes[name].get("state", "healthy") == "healthy"
            and nodes[name].get("rtt_ms") is not None
            and nodes[name].get("bandwidth_mbps") is not None
            for name in workers
        )
        if ready:
            return {name: nodes[name] for name in workers}
        if time.monotonic() > deadline:
            raise PipelineError(
                f"workers {workers} are not all connected and measured; "
                f"coordinator knows {sorted(nodes)}"
            )
        time.sleep(1)


# --------------------------------------------------------------------- the work
def deploy_stage(
    host: str, port: int, worker: str, spec: StageSpec, weights: dict[str, torch.Tensor]
) -> float:
    """Send a stage's operations and weights once; return the seconds it took."""
    started_at = time.perf_counter()
    call_worker(
        host, port, worker, "stage_define", {"stage_id": spec.stage_id},
        json.dumps(spec.to_dict(), separators=(",", ":")).encode("utf-8"),
    )
    for bundle in split_bundles(weights):
        meta, payload = encode_bundle(bundle)
        call_worker(
            host, port, worker, "stage_put", {"stage_id": spec.stage_id, "tensors": meta}, payload
        )
    return time.perf_counter() - started_at


def run_stage(
    host: str, port: int, worker: str, stage_id: str, inputs: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    output_meta, output_payload, record = run_stage_payload(
        host, port, worker, stage_id, inputs
    )
    return decode_bundle(output_meta, output_payload), record


def run_stage_payload(
    host: str, port: int, worker: str, stage_id: str, inputs: dict[str, torch.Tensor]
) -> tuple[list[dict[str, Any]], bytes, dict[str, Any]]:
    """Run a stage and return its encoded output without materializing tensors."""
    meta, payload = encode_bundle(inputs)
    started_at = time.perf_counter()
    response, reply_payload = call_worker(
        host, port, worker, "stage_run", {"stage_id": stage_id, "tensors": meta}, payload
    )
    wall = time.perf_counter() - started_at
    output_meta = response.get("tensors")
    if not isinstance(output_meta, list):
        raise PipelineError(f"{worker}: stage returned invalid tensor metadata")
    return output_meta, reply_payload, {
        "wall": wall,
        "compute": float(response.get("compute_seconds", 0.0)),
        "sent_bytes": len(payload),
        "received_bytes": len(reply_payload),
    }


@dataclass
class PipelineResult:
    cut: str
    workers: tuple[str, str]
    microbatches: int
    batch: int
    flops: tuple[int, int]
    weight_bytes: tuple[int, int]
    crossing_bytes: int
    deploy_seconds: tuple[float, float]
    run_seconds: list[float]
    calls: list[dict[str, Any]]  # from the run closest to the median
    local_seconds: dict[str, float]
    estimate_seconds: float | None
    links: dict[str, dict[str, Any]]
    max_abs_diff: float
    output: Any = field(repr=False, default=None)

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.run_seconds)

    def to_dict(self) -> dict[str, Any]:
        data = {key: value for key, value in self.__dict__.items() if key != "output"}
        data["median_seconds"] = self.median_seconds
        return data


@dataclass(frozen=True)
class _SpilledActivation:
    path: Path
    metadata: list[dict[str, Any]]


class _ActivationSpool:
    """Store queued stage activations in temporary files on the coordinator."""

    def __init__(self) -> None:
        directory = os.environ.get("FSSLK_ACTIVATION_SPOOL_DIR") or None
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="fsslk-activations-", dir=directory
        )
        self._root = Path(self._temporary_directory.name)
        self._next_id = 0

    def write(self, metadata: list[dict[str, Any]], payload: bytes) -> _SpilledActivation:
        path = self._root / f"activation-{self._next_id:08d}.bin"
        self._next_id += 1
        with path.open("wb") as handle:
            handle.write(payload)
        return _SpilledActivation(path, metadata)

    def read(self, activation: _SpilledActivation) -> dict[str, torch.Tensor]:
        try:
            payload = activation.path.read_bytes()
        finally:
            self.discard(activation)
        return decode_bundle(activation.metadata, payload)

    @staticmethod
    def discard(activation: _SpilledActivation) -> None:
        activation.path.unlink(missing_ok=True)

    def close(self) -> None:
        self._temporary_directory.cleanup()

    def __enter__(self) -> "_ActivationSpool":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _run_once(
    host: str,
    port: int,
    workers: tuple[str, str],
    specs: tuple[StageSpec, StageSpec],
    chunks: list[torch.Tensor],
    input_name: str,
    output_names: list[str],
) -> tuple[list[torch.Tensor], list[dict[str, Any]], float]:
    """Stream every micro-batch through both stages; stage A and B overlap."""
    spec_a, spec_b = specs
    # Keep at most one activation reference waiting for B. A blocks here when
    # B is slower, while the tensor bytes themselves live in the coordinator's
    # temporary spool instead of an unbounded in-memory queue.
    handoff: queue.Queue = queue.Queue(maxsize=1)
    results: dict[int, dict[str, torch.Tensor]] = {}
    calls: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    stop = threading.Event()

    def enqueue(item: tuple[int, _SpilledActivation] | None) -> bool:
        while not stop.is_set():
            try:
                handoff.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    with _ActivationSpool() as spool:
        def stage_a() -> None:
            try:
                for index, chunk in enumerate(chunks):
                    if stop.is_set():
                        break
                    feed = {item.name: chunk for item in spec_a.inputs}
                    metadata, payload, record = run_stage_payload(
                        host, port, workers[0], spec_a.stage_id, feed
                    )
                    calls.append({"stage": "A", "microbatch": index, **record})
                    activation = spool.write(metadata, payload)
                    del metadata, payload
                    if not enqueue((index, activation)):
                        spool.discard(activation)
                        break
            except BaseException as exc:  # reported to the caller after the threads end
                errors.append(exc)
            finally:
                enqueue(None)

        def stage_b() -> None:
            try:
                while (item := handoff.get()) is not None:
                    index, activation = item
                    middle = spool.read(activation)
                    feed = {
                        spec.name: middle[spec.name] if spec.name in middle else chunks[index]
                        for spec in spec_b.inputs
                    }
                    outputs, record = run_stage(host, port, workers[1], spec_b.stage_id, feed)
                    calls.append({"stage": "B", "microbatch": index, **record})
                    results[index] = outputs
                    del feed, middle
            except BaseException as exc:
                errors.append(exc)
                stop.set()

        started_at = time.perf_counter()
        threads = [threading.Thread(target=stage_a), threading.Thread(target=stage_b)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.perf_counter() - started_at
        if errors:
            raise PipelineError(str(errors[0])) from errors[0]
        outputs = [
            torch.cat([results[index][name] for index in range(len(chunks))], dim=0)
            for name in output_names
        ]
        return outputs, calls, elapsed


def _time_local(model: nn.Module, example: torch.Tensor, threads: int) -> float:
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with torch.no_grad():
            model(example)  # warm up
            samples = []
            for _ in range(3):
                started_at = time.perf_counter()
                model(example)
                samples.append(time.perf_counter() - started_at)
        return min(samples)
    finally:
        torch.set_num_threads(previous)


def run_pipeline(
    model: nn.Module,
    example_input: torch.Tensor,
    *,
    host: str = "127.0.0.1",
    port: int = 27494,
    workers: tuple[str, str] = ("worker-fast", "worker-medium"),
    cut: str | None = None,
    microbatches: int = 1,
    min_share: float = 0.2,
    repeats: int = 3,
    log: Callable[[str], None] = print,
) -> PipelineResult:
    if model.training:
        raise PipelineError("put the model in eval mode (model.eval()) before running a pipeline")
    if repeats < 1:
        raise PipelineError("repeats must be at least 1")
    if microbatches < 1 or example_input.shape[0] % microbatches:
        raise PipelineError(f"batch of {example_input.shape[0]} does not split into {microbatches} micro-batches")
    chunks = [chunk.contiguous() for chunk in example_input.chunk(microbatches)]

    # The graph has static shapes, so capture it at the micro-batch shape.
    graph = capture_graph(model, (chunks[0],))
    problems = graph.shippable_problems()
    if problems:
        raise PipelineError("model cannot be shipped to workers: " + "; ".join(problems))
    input_nodes = [node.name for node in graph.nodes if node.kind == "input"]
    if len(input_nodes) != 1:
        raise PipelineError("only models with a single tensor input are supported")
    cut_name = cut or choose_cut(graph, min_share).after
    spec_a, spec_b = split_two_stage(graph, cut_name, f"p{uuid.uuid4().hex[:8]}")
    specs = (spec_a, spec_b)

    state = model.state_dict()
    sources = {node.name: node.source for node in graph.nodes if node.kind in ("param", "buffer")}
    weights = tuple(
        {item.name: state[sources[item.name]].detach().contiguous() for item in spec.params}
        for spec in specs
    )

    log("waiting for the coordinator to measure both workers ...")
    status = wait_for_workers(host, port, list(workers))

    log("local baseline ...")
    with torch.no_grad():
        expected = model(example_input)
    expected_list = [expected] if isinstance(expected, torch.Tensor) else list(expected)
    local = {"1 thread": _time_local(model, example_input, 1)}
    default_threads = torch.get_num_threads()
    if default_threads > 1:
        local[f"{default_threads} threads"] = _time_local(model, example_input, default_threads)

    try:
        log(f"deploying stages (cut after {cut_name}) ...")
        with ThreadPoolExecutor(max_workers=2) as pool:
            deploys = [
                pool.submit(deploy_stage, host, port, workers[i], specs[i], weights[i])
                for i in range(2)
            ]
            deploy_seconds = tuple(future.result() for future in deploys)

        output_names = graph.output_names()
        log("warm-up run ...")
        _run_once(host, port, workers, specs, chunks, input_nodes[0], output_names)
        runs = []
        for number in range(repeats):
            log(f"measured run {number + 1}/{repeats} ...")
            runs.append(_run_once(host, port, workers, specs, chunks, input_nodes[0], output_names))
    finally:
        for index in range(2):
            try:
                call_worker(host, port, workers[index], "stage_drop", {"stage_id": specs[index].stage_id})
            except (PipelineError, OSError):
                pass

    times = [elapsed for _, _, elapsed in runs]
    outputs, calls, _ = min(runs, key=lambda run: abs(run[2] - statistics.median(times)))
    if len(outputs) != len(expected_list):
        raise PipelineError("pipeline returned a different number of outputs than the model")
    max_abs_diff = max(
        (got - want).abs().max().item() for got, want in zip(outputs, expected_list)
    )

    links = {name: status[name] for name in workers}
    cut_point = next(point for point in graph.cut_points() if point.after == cut_name)
    result = PipelineResult(
        cut=cut_name,
        workers=workers,
        microbatches=microbatches,
        batch=example_input.shape[0],
        flops=(stage_flops(graph, spec_a), stage_flops(graph, spec_b)),
        weight_bytes=tuple(sum(t.numel() * t.element_size() for t in w.values()) for w in weights),
        crossing_bytes=cut_point.transfer_bytes,
        deploy_seconds=deploy_seconds,
        run_seconds=times,
        calls=calls,
        local_seconds=local,
        estimate_seconds=_estimate(graph, specs, links, calls, local["1 thread"], microbatches),
        links=links,
        max_abs_diff=max_abs_diff,
        output=outputs[0] if len(outputs) == 1 else tuple(outputs),
    )
    return result


def _estimate(
    graph: CapturedGraph,
    specs: tuple[StageSpec, StageSpec],
    links: dict[str, dict[str, Any]],
    calls: list[dict[str, Any]],
    local_one_thread_seconds: float,
    microbatches: int,
) -> float | None:
    """Predict the run from the coordinator's link measurements and local FLOP rate.

    Each call pays one RTT plus all its bytes at the measured bandwidth (reduced
    by measured loss), plus compute at the single-thread local FLOP rate. With
    micro-batches the stages overlap, so the run costs the sum of one pass plus
    (micro-batches - 1) times the slowest stage.
    """
    rate = graph.total_flops * microbatches / local_one_thread_seconds
    times = []
    for stage, spec, worker in zip("AB", specs, links):
        status = links[worker]
        if status.get("bandwidth_mbps") is None or status.get("rtt_ms") is None:
            return None
        link = LinkProfile(
            worker, status["bandwidth_mbps"], status["rtt_ms"], status.get("packet_loss_pct") or 0.0
        )
        sample = next(call for call in calls if call["stage"] == stage)
        compute = stage_flops(graph, spec) / rate
        times.append(link.transfer_seconds(sample["sent_bytes"] + sample["received_bytes"]) + compute)
    return sum(times) + (microbatches - 1) * max(times)


# ----------------------------------------------------------------------- output
def _fmt_bytes(count: float) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB"):
        if value < 1024:
            return f"{value:.0f} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def format_report(result: PipelineResult) -> str:
    a, b = result.workers
    lines = [
        f"FSSLK pipeline: {a} -> {b}",
        f"cut after {result.cut} | {result.microbatches} micro-batch(es) of "
        f"{result.batch // result.microbatches} | crossing {_fmt_bytes(result.crossing_bytes)} per micro-batch",
        "",
        "stage  worker           FLOPs      weights     deploy",
    ]
    for label, worker, flops, weights, deploy in zip(
        "AB", result.workers, result.flops, result.weight_bytes, result.deploy_seconds
    ):
        lines.append(
            f"{label}      {worker:<15}  {flops / 1e9:6.1f} G   {_fmt_bytes(weights):>9}   {deploy:5.1f} s"
        )
    lines += ["", "link measured by the coordinator"]
    for worker, status in result.links.items():
        lines.append(
            f"  {worker:<15} rtt {status['rtt_ms']} ms | {status['bandwidth_mbps']} Mbps (probe) | "
            f"loss {status.get('packet_loss_pct')}%"
        )

    lines += ["", f"runs: {', '.join(f'{seconds:.2f} s' for seconds in result.run_seconds)}  (median {result.median_seconds:.2f} s)"]
    for name, seconds in result.local_seconds.items():
        lines.append(f"local, one process, {name}: {seconds:.2f} s")
    if result.estimate_seconds is not None:
        lines.append(f"predicted from probes: {result.estimate_seconds:.2f} s")

    lines += ["", "calls of the median run (wall = compute + network + relay)"]
    lines.append("stage  mb   wall   compute  other   sent       received")
    for call in sorted(result.calls, key=lambda item: (item["microbatch"], item["stage"])):
        lines.append(
            f"{call['stage']}      {call['microbatch']:<3}  {call['wall']:5.2f}  {call['compute']:7.2f}  "
            f"{call['wall'] - call['compute']:5.2f}   {_fmt_bytes(call['sent_bytes']):>9}  {_fmt_bytes(call['received_bytes']):>9}"
        )
    lines += ["", f"max |pipeline - local| = {result.max_abs_diff:.3e}"]
    return "\n".join(lines)


# -------------------------------------------------------------------------- CLI
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=27494)
    parser.add_argument(
        "--workers", default="worker-fast,worker-medium",
        help="two node ids, stage A first (default: %(default)s)",
    )
    parser.add_argument("--cut", help="cut after this node (default: cheapest balanced cut)")
    parser.add_argument("--microbatches", type=int, default=1)
    parser.add_argument("--min-share", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="also write the measurements to this file")


def main(args: argparse.Namespace) -> None:
    from .graph_demo import build_model

    workers = tuple(name.strip() for name in args.workers.split(","))
    if len(workers) != 2 or not all(workers):
        raise PipelineError("--workers needs exactly two node ids")
    torch.manual_seed(args.seed)
    model = build_model(args.layers, args.dim, args.heads)
    example = torch.randn(args.batch, args.seq, args.dim)
    result = run_pipeline(
        model, example, host=args.host, port=args.port, workers=workers,  # type: ignore[arg-type]
        cut=args.cut, microbatches=args.microbatches, min_share=args.min_share,
        repeats=args.repeats, log=lambda text: print(text, flush=True),
    )
    print()
    print(format_report(result))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=1)
        print(f"\nmeasurements written to {args.json}")
