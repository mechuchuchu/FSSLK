"""CPU worker that receives and computes small PyTorch tensor tasks."""

from __future__ import annotations

import json
import os
import re
import socket
import statistics
import threading
import time
import uuid
from typing import Any

import torch

from .bundle import decode_bundle, encode_bundle
from .linkemu import LinkEmulation
from .protocol import receive_frame, receive_message, send_frame
from .stage import StageRuntime, StageSpec
from .tensor import decode_tensor
from .udp_probe import UDP_PROBE_PACKET, decode_probe_packet, encode_probe_packet


CPU_BENCHMARK_ELEMENTS = 262_144
CPU_BENCHMARK_REPEATS = 5
UDP_PROBE_PACKET_COUNT = 100
UDP_PROBE_INTERVAL_SECONDS = 30
UDP_PROBE_RESPONSE_TIMEOUT_SECONDS = 2
MAX_STAGES_PER_WORKER = 4
_STAGE_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def _compute(message: dict[str, Any], payload: bytes) -> dict[str, Any]:
    if message.get("operation") != "sum":
        raise ValueError(f"Unsupported operation: {message.get('operation')}")
    tensor = decode_tensor(message, payload)
    if tensor.numel() == 0:
        raise ValueError("Task tensor must contain at least one element")
    return {"result": tensor.sum().item()}


def _handle_stage(
    message: dict[str, Any],
    payload: bytes,
    stages: dict[str, StageRuntime],
) -> tuple[dict[str, Any], bytes]:
    """Run one stage operation. Everything in ``message`` is untrusted input."""
    try:
        operation = message.get("op")
        stage_id = message.get("stage_id")
        if not isinstance(stage_id, str) or not _STAGE_ID.fullmatch(stage_id):
            raise ValueError("invalid stage id")

        if operation == "define":
            spec = StageSpec.from_dict(json.loads(payload.decode("utf-8")))
            if spec.stage_id != stage_id:
                raise ValueError("stage id does not match the stage definition")
            if stage_id not in stages and len(stages) >= MAX_STAGES_PER_WORKER:
                raise ValueError(f"a worker holds at most {MAX_STAGES_PER_WORKER} stages")
            stages[stage_id] = StageRuntime(spec)
            return {"ok": True, "params_expected": len(spec.params)}, b""

        runtime = stages.get(stage_id)
        if runtime is None:
            raise ValueError(f"unknown stage: {stage_id!r}")
        if operation == "put":
            runtime.put_params(decode_bundle(message.get("tensors"), payload))
            return {"ok": True, "params_missing": len(runtime.missing_params)}, b""
        if operation == "run":
            inputs = decode_bundle(message.get("tensors"), payload)
            started_at = time.perf_counter()
            outputs = runtime.run(inputs)
            compute_seconds = time.perf_counter() - started_at
            tensors, output_payload = encode_bundle(outputs)
            return (
                {"ok": True, "tensors": tensors, "compute_seconds": round(compute_seconds, 6)},
                output_payload,
            )
        if operation == "drop":
            del stages[stage_id]
            return {"ok": True}, b""
        raise ValueError(f"unknown stage operation: {operation!r}")
    except Exception as exc:  # Report the failure to the coordinator, keep the worker alive.
        return {"ok": False, "error": str(exc)}, b""


def _configure_threads() -> None:
    """Optionally pin torch's thread count (FSSLK_TORCH_THREADS), e.g. when several
    workers share one machine and should not compete for the same cores."""
    value = os.environ.get("FSSLK_TORCH_THREADS")
    if value:
        torch.set_num_threads(max(1, int(value)))


def _benchmark_cpu_speed() -> float:
    torch.set_num_threads(os.cpu_count() or 1)
    tensor = torch.arange(CPU_BENCHMARK_ELEMENTS, dtype=torch.int64, device="cpu")
    tensor.sum().item()
    samples = []
    for _ in range(3):
        started_at = time.perf_counter()
        for _ in range(CPU_BENCHMARK_REPEATS):
            tensor.sum().item()
        samples.append(time.perf_counter() - started_at)
    elapsed = max(statistics.median(samples), 0.000001)
    return CPU_BENCHMARK_ELEMENTS * CPU_BENCHMARK_REPEATS / elapsed


def run_worker(host: str, port: int, node_id: str | None = None, heartbeat_interval: float = 2.0) -> None:
    identity = node_id or os.environ.get("FSSLK_NODE_ID") or socket.gethostname() or uuid.uuid4().hex[:8]
    compute_speed = _benchmark_cpu_speed()
    _configure_threads()
    delay = 1.0
    while True:
        try:
            _run_connection(host, port, identity, heartbeat_interval, compute_speed)
            delay = 1.0
        except (ConnectionError, OSError, ValueError) as exc:
            print(f"[FSSLK] {identity}: coordinator unavailable ({exc}); retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 10.0)


def _run_connection(
    host: str,
    port: int,
    node_id: str,
    heartbeat_interval: float,
    compute_speed: float,
) -> None:
    conn = socket.create_connection((host, port), timeout=10)
    conn.settimeout(None)
    send_lock = threading.Lock()
    stop_heartbeat = threading.Event()
    link = LinkEmulation.from_env()
    stages: dict[str, StageRuntime] = {}
    if link.enabled:
        print(
            f"[FSSLK] {node_id}: emulating a link with {link.delay_ms:g} ms one-way delay"
            + (f" and {link.rate_mbps:g} Mbps" if link.rate_mbps else ""),
            flush=True,
        )

    def send(message: dict[str, Any], payload: bytes = b"") -> None:
        with send_lock:  # one frame at a time, like a real link
            link.wait(len(payload))
            send_frame(conn, message, payload)

    stream = conn.makefile("rb")
    try:
        send({
            "role": "worker",
            "node_id": node_id,
            "cpu_count": os.cpu_count() or 1,
            "device": "cpu",
            "compute_elements_per_second": compute_speed,
            "heartbeat_interval": heartbeat_interval,
        })
        acknowledgement = receive_message(stream)
        if not acknowledgement or acknowledgement.get("type") != "registered":
            raise ConnectionError("Coordinator did not acknowledge registration")
        try:
            udp_probe_token = bytes.fromhex(str(acknowledgement["udp_probe_token"]))
            udp_port = int(acknowledgement["udp_port"])
            control_token = str(acknowledgement["control_token"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConnectionError("Coordinator did not provide worker control settings") from exc
        if len(udp_probe_token) != 16 or len(control_token) != 32 or not 1 <= udp_port <= 65535:
            raise ConnectionError("Coordinator provided invalid UDP probe settings")
        print(f"[FSSLK] Worker ready: {node_id} (CPU, {os.cpu_count() or 1} cores)", flush=True)

        heartbeat = threading.Thread(
            target=_control_heartbeat_loop,
            args=(host, port, node_id, control_token, heartbeat_interval, stop_heartbeat),
            daemon=True,
        )
        heartbeat.start()
        packet_loss_monitor = threading.Thread(
            target=_udp_packet_loss_loop,
            args=(host, udp_port, udp_probe_token, send, stop_heartbeat),
            daemon=True,
        )
        packet_loss_monitor.start()

        while True:
            frame = receive_frame(stream)
            if frame is None:
                raise ConnectionError("Coordinator closed the connection")
            message, payload = frame
            link.wait(len(payload))
            message_type = message.get("type")
            if message_type == "stage":
                reply, reply_payload = _handle_stage(message, payload, stages)
                send(
                    {"type": "stage_result", "task_id": str(message.get("task_id", "")), **reply},
                    reply_payload,
                )
                continue
            if message_type == "probe":
                if payload:
                    raise ValueError("RTT probe must not contain a binary payload")
                send(
                    {
                        "type": "probe_result",
                        "probe_id": str(message.get("probe_id", "")),
                    }
                )
                continue
            if message_type == "bandwidth_probe":
                send(
                    {
                        "type": "bandwidth_result",
                        "probe_id": str(message.get("probe_id", "")),
                        "received_bytes": len(payload),
                    }
                )
                continue
            if message_type != "task":
                if payload:
                    raise ValueError("unexpected binary payload in coordinator message")
                continue
            task_id = str(message.get("task_id", ""))
            try:
                output = _compute(message, payload)
                send({"type": "result", "task_id": task_id, "ok": True, **output})
            except Exception as exc:  # Return task errors to the coordinator without killing the worker.
                send({"type": "result", "task_id": task_id, "ok": False, "error": str(exc)})
    finally:
        stop_heartbeat.set()
        try:
            stream.close()
        except OSError:
            pass
        conn.close()


def _control_heartbeat_loop(
    host: str,
    port: int,
    node_id: str,
    control_token: str,
    heartbeat_interval: float,
    stop_event: threading.Event,
) -> None:
    """Maintain an acknowledged heartbeat over a separate TCP connection."""
    interval = max(float(heartbeat_interval), 0.1)
    retry_delay = interval
    while not stop_event.is_set():
        conn: socket.socket | None = None
        stream = None
        try:
            conn = socket.create_connection((host, port), timeout=5)
            conn.settimeout(max(interval * 3, 3.0))
            stream = conn.makefile("rb")
            send_frame(
                conn,
                {
                    "role": "worker_control",
                    "node_id": node_id,
                    "control_token": control_token,
                },
            )
            acknowledgement = receive_message(stream)
            if (
                not acknowledgement
                or acknowledgement.get("type") != "control_registered"
                or acknowledgement.get("node_id") != node_id
            ):
                raise ConnectionError("Coordinator rejected the control channel")

            sequence = 0
            retry_delay = interval
            while not stop_event.wait(interval):
                sequence += 1
                send_frame(conn, {"type": "heartbeat", "sequence": sequence})
                acknowledgement = receive_message(stream)
                if (
                    not acknowledgement
                    or acknowledgement.get("type") != "heartbeat_ack"
                    or acknowledgement.get("sequence") != sequence
                ):
                    raise ConnectionError("Coordinator returned an invalid heartbeat acknowledgement")
        except (ConnectionError, OSError, ValueError) as exc:
            if not stop_event.is_set():
                print(
                    f"[FSSLK] {node_id}: control heartbeat lost ({exc}); "
                    f"reconnecting in {retry_delay:.1f}s",
                    flush=True,
                )
        finally:
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
            if conn is not None:
                conn.close()

        if stop_event.wait(retry_delay):
            return
        retry_delay = min(retry_delay * 2, 10.0)


def _udp_packet_loss_loop(
    host: str,
    port: int,
    token: bytes,
    send_control: Any,
    stop_event: threading.Event,
) -> None:
    try:
        address = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)[0][4]
    except OSError:
        return

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe_socket:
        probe_socket.settimeout(0.1)
        while not stop_event.is_set():
            probe_id = uuid.uuid4().bytes
            sent_packets = 0
            for sequence in range(UDP_PROBE_PACKET_COUNT):
                try:
                    probe_socket.sendto(
                        encode_probe_packet(token, probe_id, sequence),
                        address,
                    )
                    sent_packets += 1
                except OSError:
                    break
                if stop_event.wait(0.01):
                    return

            received_sequences: set[int] = set()
            deadline = time.monotonic() + UDP_PROBE_RESPONSE_TIMEOUT_SECONDS
            while len(received_sequences) < sent_packets and time.monotonic() < deadline:
                try:
                    packet, _ = probe_socket.recvfrom(UDP_PROBE_PACKET.size)
                except socket.timeout:
                    continue
                except OSError:
                    return
                decoded = decode_probe_packet(packet)
                if decoded is None:
                    continue
                response_token, response_probe_id, sequence = decoded
                if (
                    response_token == token
                    and response_probe_id == probe_id
                    and sequence < sent_packets
                ):
                    received_sequences.add(sequence)

            try:
                send_control(
                    {
                        "type": "packet_loss_result",
                        "probe_id": probe_id.hex(),
                        "sent_packets": sent_packets,
                        "received_packets": len(received_sequences),
                    }
                )
            except OSError:
                return
            if stop_event.wait(UDP_PROBE_INTERVAL_SECONDS):
                return
