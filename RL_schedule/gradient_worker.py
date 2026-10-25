"""HTTP worker that computes one shard's MLP gradient from data-only tensors."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import unquote

import torch
from torch.nn import functional as F

from .mlp_baseline import MemorizationMLP
from .tensor_wire import decode_float32, encode_float32

MAX_REQUEST_BYTES = 32 * 1024 * 1024
_jobs_lock = threading.Lock()
_jobs: dict[str, "_WorkerJob"] = {}


class _WorkerJob:
    def __init__(self, task_id: str, request: dict[str, Any]) -> None:
        self.task_id = task_id
        self.request = request
        self.status = "queued"
        self.progress = 0.0
        self.heartbeat_at = time.monotonic()
        self.cancel_requested = False
        self.result: dict[str, Any] | None = None
        self.error: str | None = None


def compute_gradient(
    request: dict[str, Any],
    progress_callback: Callable[[float], bool] | None = None,
) -> dict[str, Any]:
    try:
        model_version = int(request["model_version"])
        shard_id = int(request["shard_id"])
        attempt_id = int(request.get("attempt_id", 0))
        task_id = str(request["task_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("model_version, shard_id, and task_id are required") from exc
    task_parts = task_id.split(":")
    if (
        model_version < 0
        or shard_id < 0
        or attempt_id < 0
        or len(task_parts) not in {3, 4}
        or task_parts[-3:] != [str(model_version), str(shard_id), str(attempt_id)]
        or (len(task_parts) == 4 and not task_parts[0])
    ):
        raise ValueError("task identity is invalid")

    model_config = request.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("model_config is required")
    try:
        input_dim = int(model_config["input_dim"])
        hidden_dim = int(model_config["hidden_dim"])
        output_dim = int(model_config["output_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("model_config dimensions are invalid") from exc
    if min(input_dim, hidden_dim, output_dim) < 1 or max(input_dim, hidden_dim, output_dim) > 8192:
        raise ValueError("model_config dimensions are outside the supported range")

    inputs = decode_float32(request.get("inputs"))
    targets = decode_float32(request.get("targets"))
    if inputs.ndim != 2 or inputs.shape[1] != input_dim:
        raise ValueError("input tensor shape does not match model_config")
    if targets.ndim != 2 or targets.shape != (inputs.shape[0], output_dim):
        raise ValueError("target tensor shape does not match inputs and model_config")
    if inputs.shape[0] < 1:
        raise ValueError("each worker must receive at least one sample")

    model = MemorizationMLP(input_dim, hidden_dim, output_dim)
    supplied_state = request.get("state_dict")
    if not isinstance(supplied_state, dict):
        raise ValueError("state_dict is required")
    expected_state = model.state_dict()
    if set(supplied_state) != set(expected_state):
        raise ValueError("state_dict parameter names do not match the MLP")
    loaded_state = {}
    for name, expected in expected_state.items():
        value = decode_float32(supplied_state[name])
        if value.shape != expected.shape:
            raise ValueError(f"state_dict shape mismatch for {name}")
        loaded_state[name] = value
    model.load_state_dict(loaded_state, strict=True)

    batch_size = int(request.get("batch_size", 32))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parameters = tuple(model.named_parameters())
    aggregate = {name: torch.zeros_like(parameter) for name, parameter in parameters}
    sample_count = int(inputs.shape[0])
    mean_loss = 0.0
    processed = 0
    if progress_callback is not None and not progress_callback(0.0):
        return {
            "model_version": model_version,
            "shard_id": shard_id,
            "attempt_id": attempt_id,
            "task_id": task_id,
            "cancelled": True,
        }
    for start in range(0, sample_count, batch_size):
        stop = min(sample_count, start + batch_size)
        count = stop - start
        model.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(inputs[start:stop]), targets[start:stop])
        loss.backward()
        weight = count / sample_count
        for name, parameter in parameters:
            if parameter.grad is not None:
                aggregate[name].add_(parameter.grad, alpha=weight)
        mean_loss += float(loss.detach().item()) * weight
        processed = stop
        if progress_callback is not None and not progress_callback(processed / sample_count):
            return {
                "model_version": model_version,
                "shard_id": shard_id,
                "attempt_id": attempt_id,
                "task_id": task_id,
                "cancelled": True,
            }
    gradients = {
        name: encode_float32(gradient) for name, gradient in aggregate.items()
    }
    return {
        "model_version": model_version,
        "shard_id": shard_id,
        "attempt_id": attempt_id,
        "task_id": task_id,
        "samples": sample_count,
        "mean_squared_error": mean_loss,
        "gradients": gradients,
    }


def _run_job(job: _WorkerJob) -> None:
    def update_progress(progress: float) -> bool:
        with _jobs_lock:
            if job.cancel_requested:
                return False
            job.status = "running"
            job.progress = progress
            job.heartbeat_at = time.monotonic()
            return True

    try:
        result = compute_gradient(job.request, progress_callback=update_progress)
        with _jobs_lock:
            if result.get("cancelled"):
                job.status = "cancelled"
                job.progress = min(job.progress, 0.999)
            else:
                job.status = "completed"
                job.progress = 1.0
                job.result = result
            job.heartbeat_at = time.monotonic()
    except Exception as exc:
        with _jobs_lock:
            job.status = "failed"
            job.error = str(exc)
            job.heartbeat_at = time.monotonic()


class GradientRequestHandler(BaseHTTPRequestHandler):
    server_version = "RLScheduleWorker/1"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"ok": True})
            return
        prefix = "/status/"
        if self.path.startswith(prefix):
            task_id = unquote(self.path[len(prefix) :])
            with _jobs_lock:
                job = _jobs.get(task_id)
                result = None if job is None else {
                    "ok": True,
                    "task_id": task_id,
                    "status": job.status,
                    "progress": job.progress,
                    "heartbeat_age_seconds": max(
                        0.0, time.monotonic() - job.heartbeat_at
                    ),
                    "error": job.error,
                }
            if result is None:
                self._send_json(404, {"ok": False, "error": "unknown task"})
            else:
                self._send_json(200, result)
            return
        if self.path.startswith("/result/"):
            task_id = unquote(self.path[len("/result/") :])
            with _jobs_lock:
                job = _jobs.get(task_id)
                if job is None:
                    status = 404
                    result = {"ok": False, "error": "unknown task"}
                elif job.status != "completed" or job.result is None:
                    status = 409
                    result = {
                        "ok": False,
                        "status": job.status,
                        "error": job.error,
                    }
                else:
                    status = 200
                    result = {"ok": True, **job.result}
            self._send_json(status, result)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        try:
            if self.path == "/gradient":
                request = self._read_request()
                result = compute_gradient(request)
                self._send_json(200, {"ok": True, **result})
                return
            if self.path == "/submit":
                request = self._read_request()
                task_id = str(request.get("task_id", ""))
                if not task_id:
                    raise ValueError("task_id is required")
                with _jobs_lock:
                    existing = _jobs.get(task_id)
                    if existing is None:
                        job = _WorkerJob(task_id, request)
                        _jobs[task_id] = job
                if existing is not None:
                    self._send_json(
                        202,
                        {
                            "ok": True,
                            "task_id": task_id,
                            "status": existing.status,
                            "duplicate_submission": True,
                        },
                    )
                    return
                threading.Thread(target=_run_job, args=(job,), daemon=True).start()
                self._send_json(
                    202,
                    {"ok": True, "task_id": task_id, "status": "queued"},
                )
                return
            if self.path.startswith("/cancel/"):
                task_id = unquote(self.path[len("/cancel/") :])
                with _jobs_lock:
                    job = _jobs.get(task_id)
                    if job is not None and job.status in {"queued", "running"}:
                        job.cancel_requested = True
                if job is None:
                    self._send_json(404, {"ok": False, "error": "unknown task"})
                else:
                    self._send_json(200, {"ok": True, "task_id": task_id})
                return
            if self.path.startswith("/forget/"):
                task_id = unquote(self.path[len("/forget/") :])
                with _jobs_lock:
                    job = _jobs.get(task_id)
                    if job is None:
                        status = 200
                        result = {"ok": True, "forgotten": False}
                    elif job.status in {"queued", "running"}:
                        status = 409
                        result = {
                            "ok": False,
                            "error": "cannot forget a running task",
                        }
                    else:
                        del _jobs[task_id]
                        status = 200
                        result = {"ok": True, "forgotten": True}
                self._send_json(status, result)
                return
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
            return
        self.send_error(404)

    def _read_request(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > MAX_REQUEST_BYTES:
            raise ValueError("request body size is outside the supported range")
        request = json.loads(self.rfile.read(length))
        if not isinstance(request, dict):
            raise ValueError("request body must be a JSON object")
        return request

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, result: dict[str, Any]) -> None:
        body = json.dumps(result, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a shard-gradient worker.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=27500)
    parser.add_argument("--name", default="worker")
    args = parser.parse_args()
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    torch.set_num_threads(1)
    server = ThreadingHTTPServer((args.host, args.port), GradientRequestHandler)
    server.daemon_threads = True
    print(f"[{args.name}] listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
