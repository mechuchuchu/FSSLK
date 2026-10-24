"""gRPC transport spike for worker sessions, health, and pipeline RPCs."""

from __future__ import annotations

import json
import math
import os
import queue
import secrets
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator

import grpc

from .coordinator import (
    Coordinator,
    REMOTE_CALL_TIMEOUT_SECONDS,
    STAGE_COMMANDS,
)
from .protocol import (
    MAX_PAYLOAD_BYTES,
    MAX_REMOTE_CALL_BYTES,
    iter_payload_chunks,
    payload_chunk_count,
    validate_payload_chunk,
)
from .tensor import decode_tensor
from .proto import fsslk_pb2, fsslk_pb2_grpc


_MAX_GRPC_MESSAGE_BYTES = MAX_PAYLOAD_BYTES + 128 * 1024
_SESSION_QUEUE_SIZE = 32
_STREAM_CLOSED = object()


def _dump_json(value: dict[str, Any]) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _load_json(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("RPC metadata is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("RPC metadata must be a JSON object")
    return decoded


def _channel_options() -> tuple[tuple[str, int], ...]:
    return (
        ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
    )


def _reply(metadata: dict[str, Any], payload: bytes = b"") -> fsslk_pb2.RpcReply:
    return fsslk_pb2.RpcReply(metadata_json=_dump_json(metadata), payload=payload)


def _send_remote_result(
    send: Any,
    task_id: str,
    result: dict[str, Any],
    payload: bytes,
) -> None:
    if len(payload) > MAX_REMOTE_CALL_BYTES:
        result = {
            "ok": False,
            "error": f"remote result exceeds the {MAX_REMOTE_CALL_BYTES}-byte total limit",
        }
        payload = b""
    count = payload_chunk_count(len(payload))
    for index, chunk in enumerate(iter_payload_chunks(payload)):
        send(
            {
                "type": "remote_result_chunk",
                "task_id": task_id,
                "chunk_index": index,
                "chunk_count": count,
                "total_size": len(payload),
                **result,
            },
            chunk,
        )


def _append_remote_chunk(
    transfers: dict[str, dict[str, Any]],
    task_id: str,
    message: dict[str, Any],
    payload: bytes,
) -> bytes | None:
    chunk_index = message.get("chunk_index")
    chunk_count = message.get("chunk_count")
    total_size = message.get("total_size")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (chunk_index, chunk_count, total_size)
    ):
        raise ValueError("remote call chunk metadata is invalid")
    validate_payload_chunk(
        payload,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        total_size=total_size,
    )

    transfer = transfers.get(task_id)
    if transfer is None:
        if chunk_index != 0:
            raise ValueError("remote call chunks arrived out of order")
        transfer = {
            "chunk_count": chunk_count,
            "total_size": total_size,
            "chunks": [],
            "received_size": 0,
        }
        transfers[task_id] = transfer
    elif (
        transfer["chunk_count"] != chunk_count
        or transfer["total_size"] != total_size
        or chunk_index != len(transfer["chunks"])
    ):
        transfers.pop(task_id, None)
        raise ValueError("remote call chunks are inconsistent or out of order")

    transfer["chunks"].append(payload)
    transfer["received_size"] += len(payload)
    if transfer["received_size"] > MAX_REMOTE_CALL_BYTES:
        transfers.pop(task_id, None)
        raise ValueError("remote call exceeds the configured total limit")
    if len(transfer["chunks"]) != chunk_count:
        return None

    transfers.pop(task_id, None)
    if transfer["received_size"] != total_size:
        raise ValueError("remote call size does not match its metadata")
    return b"".join(transfer["chunks"])


class CoordinatorGrpcService(fsslk_pb2_grpc.FSSLKServicer):
    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator

    def WorkerSession(
        self,
        request_iterator: Iterator[fsslk_pb2.WorkerEvent],
        context: grpc.ServicerContext,
    ) -> Iterator[fsslk_pb2.CoordinatorEvent]:
        events = iter(request_iterator)
        first = next(events, None)
        if first is None or first.WhichOneof("event") != "registration":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "worker registration must start the stream")
        registration = first.registration
        node_id = registration.node_id.strip()
        if not node_id or len(node_id) > 128:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "worker node id is invalid")
        if not math.isfinite(registration.compute_elements_per_second):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "worker compute speed is invalid")

        outgoing: queue.Queue[tuple[dict[str, Any], bytes] | object] = queue.Queue(
            maxsize=_SESSION_QUEUE_SIZE
        )
        incoming: queue.Queue[fsslk_pb2.WorkerReply | object] = queue.Queue(
            maxsize=_SESSION_QUEUE_SIZE
        )
        session_closed = threading.Event()

        def send_to_worker(message: dict[str, Any], payload: bytes) -> None:
            if session_closed.is_set():
                raise OSError("worker gRPC session is closed")
            try:
                outgoing.put((dict(message), payload), timeout=1)
            except queue.Full as exc:
                raise OSError("worker gRPC command queue is full") from exc

        def close_session() -> None:
            session_closed.set()

        node = self.coordinator._register_stream_worker(
            node_id=node_id,
            cpu_count=registration.cpu_count,
            device=registration.device,
            compute_speed=registration.compute_elements_per_second,
            heartbeat_interval=registration.heartbeat_interval,
            sender=send_to_worker,
            close_sender=close_session,
        )

        def read_replies() -> None:
            try:
                for event in events:
                    if event.WhichOneof("event") != "reply":
                        continue
                    incoming.put(event.reply)
            finally:
                incoming.put(_STREAM_CLOSED)

        threading.Thread(target=read_replies, daemon=True).start()
        print(f"[FSSLK] Worker joined over gRPC: {node_id}", flush=True)
        yield fsslk_pb2.CoordinatorEvent(
            registered=fsslk_pb2.WorkerRegistered(
                node_id=node.node_id,
                udp_probe_token=node.udp_probe_token,
                udp_port=self.coordinator.udp_port,
                control_token=node.control_token,
            )
        )

        try:
            while context.is_active() and not session_closed.is_set():
                for _ in range(_SESSION_QUEUE_SIZE):
                    try:
                        reply = incoming.get_nowait()
                    except queue.Empty:
                        break
                    if reply is _STREAM_CLOSED:
                        return
                    metadata = _load_json(reply.metadata_json)
                    message = {
                        "type": reply.type,
                        "task_id": reply.task_id,
                        **metadata,
                    }
                    self.coordinator._process_worker_message(node, message, reply.payload)

                try:
                    message, payload = outgoing.get(timeout=0.1)
                except queue.Empty:
                    continue
                message_type = str(message.pop("type", ""))
                task_id = str(message.pop("task_id", ""))
                yield fsslk_pb2.CoordinatorEvent(
                    command=fsslk_pb2.WorkerCommand(
                        type=message_type,
                        task_id=task_id,
                        metadata_json=_dump_json(message),
                        payload=payload,
                    )
                )
        except (ValueError, OSError) as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        finally:
            session_closed.set()
            self.coordinator._unregister_worker(node)

    def WorkerHealth(
        self,
        request_iterator: Iterator[fsslk_pb2.WorkerHeartbeat],
        context: grpc.ServicerContext,
    ) -> Iterator[fsslk_pb2.HeartbeatAck]:
        heartbeats = iter(request_iterator)
        first = next(heartbeats, None)
        if first is None or first.sequence != 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "health stream must begin with sequence zero")

        with self.coordinator.nodes_lock:
            node = self.coordinator.nodes.get(first.node_id)
        if (
            node is None
            or not node.connected
            or not secrets.compare_digest(node.control_token, first.control_token)
        ):
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "worker health credentials are invalid")

        control_session = object()
        with node.liveness_lock:
            node.control_conn = control_session
            node.control_connected = True
            node.last_heartbeat = time.monotonic()
            node.last_heartbeat_sequence = 0
        yield fsslk_pb2.HeartbeatAck(sequence=0)
        try:
            for heartbeat in heartbeats:
                if heartbeat.node_id != node.node_id or not secrets.compare_digest(
                    node.control_token, heartbeat.control_token
                ):
                    context.abort(grpc.StatusCode.PERMISSION_DENIED, "worker health credentials changed")
                with node.liveness_lock:
                    if node.control_conn is not control_session:
                        return
                    if heartbeat.sequence <= node.last_heartbeat_sequence:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "heartbeat sequence did not advance")
                    node.last_heartbeat_sequence = heartbeat.sequence
                    node.last_heartbeat = time.monotonic()
                yield fsslk_pb2.HeartbeatAck(sequence=heartbeat.sequence)
        finally:
            with node.liveness_lock:
                if node.control_conn is control_session:
                    node.control_conn = None
                    node.control_connected = False

    def Status(self, request: fsslk_pb2.Empty, context: grpc.ServicerContext) -> fsslk_pb2.RpcReply:
        return _reply({"ok": True, "nodes": self.coordinator.status_nodes()})

    def Sum(self, request: fsslk_pb2.RpcRequest, context: grpc.ServicerContext) -> fsslk_pb2.RpcReply:
        try:
            values = decode_tensor(_load_json(request.metadata_json), request.payload)
            if values.numel() == 0:
                raise ValueError("tensor must contain at least one element")
            result, workers = self.coordinator._distributed_sum(values)
            return _reply({"ok": True, "result": result, "workers": workers})
        except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
            return _reply({"ok": False, "error": str(exc)})

    def Stage(
        self,
        request: fsslk_pb2.StageRequest,
        context: grpc.ServicerContext,
    ) -> fsslk_pb2.RpcReply:
        try:
            if request.command not in STAGE_COMMANDS:
                raise ValueError(f"unknown stage command: {request.command!r}")
            response, payload = self.coordinator.call_stage(
                request.worker_id,
                request.command,
                _load_json(request.metadata_json),
                request.payload,
            )
            return _reply(response, payload)
        except (ValueError, RuntimeError, TimeoutError, ConnectionError, OSError) as exc:
            return _reply({"ok": False, "error": str(exc)})

    def Execute(
        self,
        request_iterator: Iterator[fsslk_pb2.RemoteCallRequest],
        context: grpc.ServicerContext,
    ) -> Iterator[fsslk_pb2.RemoteCallReply]:
        requests = iter(request_iterator)
        first = next(requests, None)
        if first is None:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "remote call has no payload chunks")
        expected_token = os.environ.get("FSSLK_DDP_TOKEN")
        if not expected_token:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "remote Python calls are disabled; configure FSSLK_DDP_TOKEN",
            )
        if not secrets.compare_digest(expected_token, first.token):
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "FSSLK_DDP_TOKEN is invalid")

        try:
            validate_payload_chunk(
                first.payload,
                chunk_index=first.chunk_index,
                chunk_count=first.chunk_count,
                total_size=first.total_size,
            )
            if first.chunk_index != 0:
                raise ValueError("remote call must start with chunk zero")
            payload_parts = [first.payload]
            received_size = len(first.payload)
            timeout = first.timeout_seconds or REMOTE_CALL_TIMEOUT_SECONDS
            worker_id = first.worker_id
            chunk_count = first.chunk_count

            for expected_index, request in enumerate(requests, start=1):
                if (
                    request.worker_id != worker_id
                    or request.timeout_seconds != first.timeout_seconds
                    or request.chunk_count != chunk_count
                    or request.total_size != first.total_size
                    or not secrets.compare_digest(expected_token, request.token)
                ):
                    raise ValueError("remote call chunk metadata changed during upload")
                validate_payload_chunk(
                    request.payload,
                    chunk_index=request.chunk_index,
                    chunk_count=request.chunk_count,
                    total_size=request.total_size,
                )
                if request.chunk_index != expected_index:
                    raise ValueError("remote call chunks arrived out of order")
                payload_parts.append(request.payload)
                received_size += len(request.payload)

            if len(payload_parts) != chunk_count or received_size != first.total_size:
                raise ValueError("remote call upload ended before all chunks arrived")
            payload = b"".join(payload_parts)
            payload_parts.clear()
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        try:
            result = self.coordinator.execute_remote(
                payload,
                worker_id=worker_id or None,
                timeout=timeout,
            )
            count = payload_chunk_count(len(result))
            for index, chunk in enumerate(iter_payload_chunks(result)):
                yield fsslk_pb2.RemoteCallReply(
                    ok=True,
                    payload=chunk,
                    chunk_index=index,
                    chunk_count=count,
                    total_size=len(result),
                )
        except (ValueError, RuntimeError, TimeoutError, ConnectionError, OSError) as exc:
            yield fsslk_pb2.RemoteCallReply(
                ok=False,
                error=str(exc),
                chunk_index=0,
                chunk_count=1,
                total_size=0,
            )


def serve_grpc_forever(coordinator: Coordinator) -> None:
    """Serve gRPC APIs and retain the existing UDP link probes."""
    udp_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp_server.bind((coordinator.host, coordinator.udp_port))
    coordinator.udp_port = udp_server.getsockname()[1]
    coordinator.udp_server = udp_server

    threading.Thread(target=coordinator._network_monitor_loop, daemon=True).start()
    threading.Thread(target=coordinator._liveness_monitor_loop, daemon=True).start()
    threading.Thread(
        target=coordinator._udp_echo_loop,
        args=(udp_server,),
        daemon=True,
    ).start()

    server = grpc.server(
        ThreadPoolExecutor(max_workers=32),
        options=_channel_options(),
    )
    fsslk_pb2_grpc.add_FSSLKServicer_to_server(CoordinatorGrpcService(coordinator), server)
    if not server.add_insecure_port(f"{coordinator.host}:{coordinator.port}"):
        udp_server.close()
        raise OSError(f"could not bind gRPC port {coordinator.host}:{coordinator.port}")
    print(
        f"[FSSLK] Coordinator listening on gRPC {coordinator.host}:{coordinator.port} "
        f"and UDP {coordinator.host}:{coordinator.udp_port}",
        flush=True,
    )
    server.start()
    try:
        server.wait_for_termination()
    finally:
        server.stop(grace=2)
        udp_server.close()


def grpc_request(
    host: str,
    port: int,
    request: dict[str, Any],
    payload: bytes = b"",
    timeout: float = 900,
) -> tuple[dict[str, Any], bytes]:
    """Make a status, sum, or stage call using the gRPC client API."""
    command = request.get("command")
    metadata = {key: value for key, value in request.items() if key not in {"command", "role"}}
    try:
        with grpc.insecure_channel(f"{host}:{port}", options=_channel_options()) as channel:
            stub = fsslk_pb2_grpc.FSSLKStub(channel)
            if command == "status":
                response = stub.Status(fsslk_pb2.Empty(), timeout=timeout)
            elif command == "sum":
                response = stub.Sum(
                    fsslk_pb2.RpcRequest(metadata_json=_dump_json(metadata), payload=payload),
                    timeout=timeout,
                )
            elif isinstance(command, str) and command in STAGE_COMMANDS:
                worker_id = metadata.pop("worker", "")
                response = stub.Stage(
                    fsslk_pb2.StageRequest(
                        worker_id=worker_id,
                        command=command,
                        metadata_json=_dump_json(metadata),
                        payload=payload,
                    ),
                    timeout=timeout,
                )
            else:
                raise ValueError(f"unsupported gRPC command: {command!r}")
    except grpc.RpcError as exc:
        details = exc.details() or exc.code().name
        raise ConnectionError(f"gRPC {command} request failed: {details}") from exc
    return _load_json(response.metadata_json), response.payload


def execute_remote_request(
    host: str,
    port: int,
    worker_id: str | None,
    token: str,
    timeout_seconds: int,
    payload: bytes,
    rpc_timeout: float,
) -> fsslk_pb2.RemoteCallReply:
    """Stream a serialized callable request and reassemble its result."""
    try:
        with grpc.insecure_channel(f"{host}:{port}", options=_channel_options()) as channel:
            stub = fsslk_pb2_grpc.FSSLKStub(channel)

            count = payload_chunk_count(len(payload))

            def request_chunks() -> Iterator[fsslk_pb2.RemoteCallRequest]:
                for index, chunk in enumerate(iter_payload_chunks(payload)):
                    yield fsslk_pb2.RemoteCallRequest(
                        worker_id=worker_id or "",
                        token=token,
                        timeout_seconds=timeout_seconds,
                        payload=chunk,
                        chunk_index=index,
                        chunk_count=count,
                        total_size=len(payload),
                    )

            responses = iter(stub.Execute(request_chunks(), timeout=rpc_timeout))
            first = next(responses, None)
            if first is None:
                raise ConnectionError("Coordinator returned no remote-call result")
            if not first.ok:
                return first

            try:
                validate_payload_chunk(
                    first.payload,
                    chunk_index=first.chunk_index,
                    chunk_count=first.chunk_count,
                    total_size=first.total_size,
                )
                if first.chunk_index != 0:
                    raise ValueError("remote result must start with chunk zero")
                result_parts = [first.payload]
                result_size = len(first.payload)
                for expected_index, response in enumerate(responses, start=1):
                    if (
                        not response.ok
                        or response.chunk_count != first.chunk_count
                        or response.total_size != first.total_size
                    ):
                        raise ValueError("remote result chunk metadata changed")
                    validate_payload_chunk(
                        response.payload,
                        chunk_index=response.chunk_index,
                        chunk_count=response.chunk_count,
                        total_size=response.total_size,
                    )
                    if response.chunk_index != expected_index:
                        raise ValueError("remote result chunks arrived out of order")
                    result_parts.append(response.payload)
                    result_size += len(response.payload)
                if len(result_parts) != first.chunk_count or result_size != first.total_size:
                    raise ValueError("remote result stream ended before all chunks arrived")
            except ValueError as exc:
                raise ConnectionError(f"Coordinator returned an invalid remote result: {exc}") from exc
            return fsslk_pb2.RemoteCallReply(ok=True, payload=b"".join(result_parts))
    except grpc.RpcError as exc:
        details = exc.details() or exc.code().name
        raise ConnectionError(f"gRPC remote call failed: {details}") from exc


def run_worker_grpc(
    host: str,
    port: int,
    node_id: str | None = None,
    heartbeat_interval: float = 2.0,
) -> None:
    """Run a CPU worker over one bidirectional data stream and health stream."""
    from .worker import (
        _benchmark_cpu_speed,
        _compute,
        _configure_threads,
        _handle_stage,
        _udp_packet_loss_loop,
    )
    from .linkemu import LinkEmulation

    identity = (
        node_id
        or os.environ.get("FSSLK_NODE_ID")
        or socket.gethostname()
        or uuid.uuid4().hex[:8]
    )
    compute_speed = _benchmark_cpu_speed()
    _configure_threads()
    delay = 1.0
    while True:
        try:
            _run_grpc_worker_session(
                host,
                port,
                identity,
                heartbeat_interval,
                compute_speed,
                _compute,
                _handle_stage,
                _udp_packet_loss_loop,
                LinkEmulation,
            )
            delay = 1.0
        except (grpc.RpcError, ConnectionError, OSError, ValueError) as exc:
            print(
                f"[FSSLK] {identity}: gRPC coordinator unavailable ({exc}); retrying in {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)
            delay = min(delay * 2, 10.0)


def _run_grpc_worker_session(
    host: str,
    port: int,
    node_id: str,
    heartbeat_interval: float,
    compute_speed: float,
    compute: Any,
    handle_stage: Any,
    udp_packet_loss_loop: Any,
    link_emulation_type: Any,
) -> None:
    channel = grpc.insecure_channel(f"{host}:{port}", options=_channel_options())
    stub = fsslk_pb2_grpc.FSSLKStub(channel)
    outgoing: queue.Queue[fsslk_pb2.WorkerEvent | object] = queue.Queue(
        maxsize=_SESSION_QUEUE_SIZE
    )
    stop = threading.Event()
    stages: dict[str, Any] = {}
    remote_payloads: dict[str, dict[str, Any]] = {}
    link = link_emulation_type.from_env()

    def send(message: dict[str, Any], payload: bytes = b"") -> None:
        link.wait(len(payload))
        data = dict(message)
        message_type = str(data.pop("type", ""))
        task_id = str(data.pop("task_id", ""))
        event = fsslk_pb2.WorkerEvent(
            reply=fsslk_pb2.WorkerReply(
                type=message_type,
                task_id=task_id,
                metadata_json=_dump_json(data),
                payload=payload,
            )
        )
        try:
            outgoing.put(event, timeout=1)
        except queue.Full as exc:
            raise OSError("coordinator gRPC stream is backlogged") from exc

    def request_stream() -> Iterator[fsslk_pb2.WorkerEvent]:
        yield fsslk_pb2.WorkerEvent(
            registration=fsslk_pb2.WorkerRegistration(
                node_id=node_id,
                cpu_count=os_cpu_count(),
                device="cpu",
                compute_elements_per_second=compute_speed,
                heartbeat_interval=heartbeat_interval,
            )
        )
        while not stop.is_set():
            try:
                event = outgoing.get(timeout=0.1)
            except queue.Empty:
                continue
            if event is _STREAM_CLOSED:
                return
            yield event

    call = stub.WorkerSession(request_stream())
    first = next(call, None)
    if first is None or first.WhichOneof("event") != "registered":
        raise ConnectionError("Coordinator did not acknowledge gRPC worker registration")
    registration = first.registered
    if (
        registration.node_id != node_id
        or len(registration.udp_probe_token) != 16
        or not 1 <= registration.udp_port <= 65535
        or len(registration.control_token) != 32
    ):
        raise ConnectionError("Coordinator returned invalid gRPC worker settings")

    def heartbeat_loop() -> None:
        interval = max(float(heartbeat_interval), 0.1)
        retry_delay = interval
        while not stop.is_set():
            def heartbeats() -> Iterator[fsslk_pb2.WorkerHeartbeat]:
                yield fsslk_pb2.WorkerHeartbeat(
                    node_id=node_id,
                    control_token=registration.control_token,
                    sequence=0,
                )
                sequence = 0
                while not stop.wait(interval):
                    sequence += 1
                    yield fsslk_pb2.WorkerHeartbeat(
                        node_id=node_id,
                        control_token=registration.control_token,
                        sequence=sequence,
                    )

            try:
                expected_sequence = 0
                acknowledgements = stub.WorkerHealth(heartbeats())
                for acknowledgement in acknowledgements:
                    if acknowledgement.sequence != expected_sequence:
                        raise ConnectionError("Coordinator returned an invalid heartbeat acknowledgement")
                    expected_sequence += 1
                if not stop.is_set():
                    raise ConnectionError("Coordinator closed the gRPC health stream")
                return
            except grpc.RpcError as exc:
                if stop.is_set():
                    return
                print(
                    f"[FSSLK] {node_id}: gRPC health stream lost ({exc}); "
                    f"reconnecting in {retry_delay:.1f}s",
                    flush=True,
                )
            except ConnectionError as exc:
                if stop.is_set():
                    return
                print(f"[FSSLK] {node_id}: {exc}; reconnecting in {retry_delay:.1f}s", flush=True)
            if stop.wait(retry_delay):
                return
            retry_delay = min(retry_delay * 2, 10.0)

    threading.Thread(target=heartbeat_loop, daemon=True).start()
    threading.Thread(
        target=udp_packet_loss_loop,
        args=(
            host,
            registration.udp_port,
            registration.udp_probe_token,
            send,
            stop,
        ),
        daemon=True,
    ).start()

    try:
        for event in call:
            if event.WhichOneof("event") != "command":
                continue
            command = event.command
            message = {
                "type": command.type,
                "task_id": command.task_id,
                **_load_json(command.metadata_json),
            }
            payload = command.payload
            link.wait(len(payload))
            if command.type == "stage":
                response, response_payload = handle_stage(message, payload, stages)
                send(
                    {"type": "stage_result", "task_id": command.task_id, **response},
                    response_payload,
                )
            elif command.type == "probe":
                send({"type": "probe_result", "probe_id": str(message.get("probe_id", ""))})
            elif command.type == "bandwidth_probe":
                send(
                    {
                        "type": "bandwidth_result",
                        "probe_id": str(message.get("probe_id", "")),
                        "received_bytes": len(payload),
                    }
                )
            elif command.type == "task":
                task_id = command.task_id
                operation = message.get("operation")
                if operation == "python_fragment":
                    from .ddp import _execute_payload

                    try:
                        complete_payload = _append_remote_chunk(
                            remote_payloads,
                            task_id,
                            message,
                            payload,
                        )
                    except ValueError as exc:
                        remote_payloads.pop(task_id, None)
                        _send_remote_result(
                            send,
                            task_id,
                            {"ok": False, "error": str(exc)},
                            b"",
                        )
                        continue
                    if complete_payload is None:
                        continue
                    result, result_payload = _execute_payload(complete_payload)
                    _send_remote_result(send, task_id, result, result_payload)
                elif operation == "python":
                    # Accept one-frame calls from older coordinator versions.
                    from .ddp import _execute_payload

                    result, result_payload = _execute_payload(payload)
                    _send_remote_result(send, task_id, result, result_payload)
                else:
                    try:
                        result = compute(message, payload)
                        send({"type": "result", "task_id": task_id, "ok": True, **result})
                    except Exception as exc:
                        send({"type": "result", "task_id": task_id, "ok": False, "error": str(exc)})
    finally:
        stop.set()
        try:
            outgoing.put_nowait(_STREAM_CLOSED)
        except queue.Full:
            pass
        call.cancel()
        channel.close()


def os_cpu_count() -> int:
    return os.cpu_count() or 1
