"""Remote and axis-sharded PyTorch callable dispatch."""

from __future__ import annotations

import functools
import math
import os
import random
import sys
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import cloudpickle

from .protocol import MAX_PAYLOAD_BYTES

_CLOUDPICKLE_LOCK = threading.RLock()
_SHARDS_PER_WORKER = 4


class RemoteExecutionError(RuntimeError):
    """A callable raised an exception on its FSSLK worker."""


@dataclass(frozen=True)
class _RemoteOptions:
    host: str | None
    port: int | None
    worker: str | None
    token: str | None
    timeout: float
    axis: int | None
    shard_arg: int
    merge: str

    def resolved(self) -> tuple[str, int, str | None, str, float]:
        host = self.host or os.environ.get("FSSLK_COORDINATOR_HOST", "127.0.0.1")
        port = self.port or int(os.environ.get("FSSLK_COORDINATOR_PORT", "27494"))
        token = self.token if self.token is not None else os.environ.get("FSSLK_DDP_TOKEN")
        if not token:
            raise ValueError(
                "remote code execution requires FSSLK_DDP_TOKEN on the client"
            )
        if not 0 < self.timeout <= 3600:
            raise ValueError("timeout must be between 0 and 3600 seconds")
        return host, port, self.worker, token, self.timeout


class RemoteCallable:
    """Callable proxy that runs the wrapped Python callable on one worker."""

    def __init__(self, target: Callable[..., Any], options: _RemoteOptions):
        if not callable(target):
            raise TypeError("fsslk_ddp target must be callable")
        self._target = target
        self._options = options
        if hasattr(target, "__name__"):
            self.__name__ = target.__name__
            self.__doc__ = target.__doc__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _dispatch(
            {"kind": "call", "callable": self._target, "args": args, "kwargs": kwargs},
            self._options,
        )


class RemoteClassFactory:
    """Factory for remote callable classes; a fresh instance runs per call."""

    def __init__(self, target: type[Any], options: _RemoteOptions):
        self._target = target
        self._options = options
        self.__name__ = target.__name__
        self.__doc__ = target.__doc__

    def __call__(self, *constructor_args: Any, **constructor_kwargs: Any) -> "RemoteClassInstance":
        return RemoteClassInstance(
            self._target,
            constructor_args,
            constructor_kwargs,
            self._options,
        )


class RemoteClassInstance:
    """Configured class proxy. Its worker-side instance is recreated per call."""

    def __init__(
        self,
        target: type[Any],
        constructor_args: tuple[Any, ...],
        constructor_kwargs: dict[str, Any],
        options: _RemoteOptions,
    ):
        self._target = target
        self._constructor_args = constructor_args
        self._constructor_kwargs = constructor_kwargs
        self._options = options

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _dispatch(
            {
                "kind": "class_call",
                "class": self._target,
                "constructor_args": self._constructor_args,
                "constructor_kwargs": self._constructor_kwargs,
                "args": args,
                "kwargs": kwargs,
            },
            self._options,
        )


def fsslk_ddp(
    target: Callable[..., Any] | type[Any] | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    worker: str | None = None,
    token: str | None = None,
    timeout: float = 600,
    axis: int | None = None,
    shard_arg: int = 0,
    merge: str = "concat",
) -> Any:
    """Wrap a Python callable or class for execution on FSSLK Workers.

    With a class, calling the returned factory records constructor arguments;
    the class is instantiated on a worker for each subsequent invocation.
    Without ``worker``, invocations are assigned round-robin among healthy
    workers. The coordinator address and token can be supplied through
    ``FSSLK_COORDINATOR_HOST``, ``FSSLK_COORDINATOR_PORT``, and
    ``FSSLK_DDP_TOKEN``. With ``axis`` set, the positional tensor argument
    selected by ``shard_arg`` is split across healthy workers; ``merge`` is
    ``concat``, ``sum``, or ``list``. A pinned ``worker`` restricts that split
    to the selected Worker. Sharded work is split into up to four chunks per
    worker; a worker gets another chunk after it finishes one.
    """
    if axis is not None and (not isinstance(axis, int) or isinstance(axis, bool)):
        raise TypeError("axis must be an integer or None")
    if not isinstance(shard_arg, int) or isinstance(shard_arg, bool):
        raise TypeError("shard_arg must be an integer positional argument index")
    if merge not in {"concat", "sum", "list"}:
        raise ValueError("merge must be 'concat', 'sum', or 'list'")
    options = _RemoteOptions(host, port, worker, token, timeout, axis, shard_arg, merge)
    if target is None:
        return lambda decorated: fsslk_ddp(
            decorated,
            host=host,
            port=port,
            worker=worker,
            token=token,
            timeout=timeout,
            axis=axis,
            shard_arg=shard_arg,
            merge=merge,
        )
    if isinstance(target, type):
        return RemoteClassFactory(target, options)
    if not callable(target):
        raise TypeError("fsslk_ddp target must be a function or callable class")
    proxy = RemoteCallable(target, options)
    if hasattr(target, "__name__"):
        functools.update_wrapper(proxy, target)
    return proxy


def _dispatch(job: dict[str, Any], options: _RemoteOptions) -> Any:
    host, port, worker, token, timeout = options.resolved()
    if options.axis is not None:
        return _dispatch_sharded(job, options, host, port, worker, token, timeout)
    return _invoke_remote_job(job, host, port, worker, token, timeout)


def _invoke_remote_job(
    job: dict[str, Any],
    host: str,
    port: int,
    worker: str | None,
    token: str,
    timeout: float,
) -> Any:
    payload = _serialize_job(job)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"remote call exceeds the {MAX_PAYLOAD_BYTES}-byte payload limit")

    from .grpc_transport import execute_remote_request

    response = execute_remote_request(
        host,
        port,
        worker,
        token,
        math.ceil(timeout),
        payload,
        timeout + 10,
    )
    if not response.ok:
        raise RemoteExecutionError(response.error or "remote callable failed")
    try:
        return cloudpickle.loads(response.payload)
    except Exception as exc:
        raise RemoteExecutionError(f"worker returned an invalid result: {exc}") from exc


def _dispatch_sharded(
    job: dict[str, Any],
    options: _RemoteOptions,
    host: str,
    port: int,
    worker: str | None,
    token: str,
    timeout: float,
) -> Any:
    import torch

    args = job.get("args")
    if not isinstance(args, tuple):
        raise ValueError("sharded calls require positional arguments")
    index = options.shard_arg
    if index < 0:
        index += len(args)
    if not 0 <= index < len(args):
        raise ValueError(f"shard_arg {options.shard_arg} does not select a positional argument")
    tensor = args[index]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("the selected shard_arg must be a torch.Tensor")
    if tensor.device.type != "cpu":
        raise ValueError("FSSLK axis sharding currently supports CPU tensors only")
    if tensor.ndim == 0:
        raise ValueError("cannot split a scalar tensor across an axis")
    axis = options.axis
    assert axis is not None
    if not -tensor.ndim <= axis < tensor.ndim:
        raise ValueError(f"axis {axis} is out of range for a {tensor.ndim}-dimensional tensor")
    axis %= tensor.ndim
    extent = tensor.shape[axis]
    if extent < 1:
        raise ValueError("cannot distribute an empty tensor axis")

    from .grpc_transport import grpc_request

    status, _ = grpc_request(host, port, {"command": "status"}, timeout=timeout)
    workers = [
        node
        for node in status.get("nodes", [])
        if node.get("state") == "healthy"
        and (worker is None or node.get("node_id") == worker)
    ]
    if not workers:
        raise RuntimeError("No active workers are available for this sharded call")
    shard_count = min(extent, len(workers) * _SHARDS_PER_WORKER)
    base_count, remainder = divmod(extent, shard_count)
    shards: list[tuple[int, dict[str, Any]]] = []
    offset = 0
    for shard_index in range(shard_count):
        count = base_count + (1 if shard_index < remainder else 0)
        chunk = tensor.narrow(axis, offset, count).contiguous()
        shard_args = list(args)
        shard_args[index] = chunk
        shard_job = {**job, "args": tuple(shard_args)}
        shards.append((shard_index, shard_job))
        offset += count
    if offset != extent:
        raise RuntimeError("shards did not cover the full tensor axis")

    random.shuffle(shards)
    results: list[Any] = [None] * shard_count
    next_shard = 0
    shard_lock = threading.Lock()

    def consume_shards(node_id: str) -> None:
        nonlocal next_shard
        while True:
            with shard_lock:
                if next_shard >= len(shards):
                    return
                shard_index, shard_job = shards[next_shard]
                next_shard += 1
            results[shard_index] = _invoke_remote_job(
                shard_job, host, port, node_id, token, timeout
            )

    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = [pool.submit(consume_shards, node["node_id"]) for node in workers]
        for future in futures:
            future.result()
    return _merge_results(results, options.merge, axis)


def _merge_results(results: list[Any], merge: str, axis: int) -> Any:
    import torch

    if merge == "list":
        return results
    if all(isinstance(result, torch.Tensor) for result in results):
        if merge == "concat":
            return torch.cat(results, dim=axis)
        return torch.stack(results).sum(dim=0)
    if all(isinstance(result, dict) for result in results):
        keys = set(results[0])
        if any(set(result) != keys for result in results):
            raise ValueError("shard results returned different dictionary keys")
        return {
            key: _merge_results([result[key] for result in results], merge, axis)
            for key in results[0]
        }
    if all(isinstance(result, (tuple, list)) for result in results):
        size = len(results[0])
        if any(len(result) != size for result in results):
            raise ValueError("shard results returned different sequence lengths")
        values = [
            _merge_results([result[index] for result in results], merge, axis)
            for index in range(size)
        ]
        return tuple(values) if isinstance(results[0], tuple) else values
    if merge == "sum" and all(isinstance(result, (int, float)) for result in results):
        return sum(results)
    raise TypeError(f"merge={merge!r} cannot combine {type(results[0]).__name__} results")


def _execute_payload(payload: bytes) -> tuple[dict[str, Any], bytes]:
    """Worker entry point. Payloads must originate from an authenticated client."""
    try:
        job = cloudpickle.loads(payload)
        if not isinstance(job, dict):
            raise ValueError("remote call description must be a dictionary")
        args = job.get("args", ())
        kwargs = job.get("kwargs", {})
        if not isinstance(args, tuple) or not isinstance(kwargs, dict):
            raise ValueError("remote call arguments are malformed")
        kind = job.get("kind")
        if kind == "call":
            target = job.get("callable")
            if not callable(target):
                raise ValueError("remote target is not callable")
        elif kind == "class_call":
            target_type = job.get("class")
            constructor_args = job.get("constructor_args", ())
            constructor_kwargs = job.get("constructor_kwargs", {})
            if (
                not isinstance(target_type, type)
                or not isinstance(constructor_args, tuple)
                or not isinstance(constructor_kwargs, dict)
            ):
                raise ValueError("remote class construction data is malformed")
            target = target_type(*constructor_args, **constructor_kwargs)
        else:
            raise ValueError(f"unknown remote call kind: {kind!r}")

        result_payload = cloudpickle.dumps(
            target(*args, **kwargs), protocol=cloudpickle.DEFAULT_PROTOCOL
        )
        if len(result_payload) > MAX_PAYLOAD_BYTES:
            raise ValueError(f"remote result exceeds the {MAX_PAYLOAD_BYTES}-byte payload limit")
        return {"ok": True}, result_payload
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, b""


def _serialize_job(job: dict[str, Any]) -> bytes:
    target = job.get("callable") if "callable" in job else job.get("class")
    if isinstance(target, type):
        module_name = target.__module__
    else:
        module_name = getattr(target, "__module__", type(target).__module__)
    module = sys.modules.get(module_name)
    # cloudpickle normally refers to importable top-level functions by module
    # name. Send their defining module by value so Workers receive the code.
    if (
        module is None
        or module_name == "__main__"
        or module_name in sys.stdlib_module_names
        or module_name.startswith(("torch", "fsslk"))
    ):
        return cloudpickle.dumps(job, protocol=cloudpickle.DEFAULT_PROTOCOL)
    with _CLOUDPICKLE_LOCK:
        try:
            cloudpickle.register_pickle_by_value(module)
        except (ImportError, ValueError):
            return cloudpickle.dumps(job, protocol=cloudpickle.DEFAULT_PROTOCOL)
        try:
            return cloudpickle.dumps(job, protocol=cloudpickle.DEFAULT_PROTOCOL)
        finally:
            cloudpickle.unregister_pickle_by_value(module)
