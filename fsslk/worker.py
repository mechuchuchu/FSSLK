"""CPU worker that receives and computes small PyTorch tensor tasks."""

from __future__ import annotations

import os
import socket
import statistics
import threading
import time
import uuid
from typing import Any

import torch

from .protocol import receive_frame, receive_message, send_frame
from .tensor import decode_tensor
from .udp_probe import UDP_PROBE_PACKET, decode_probe_packet, encode_probe_packet


CPU_BENCHMARK_ELEMENTS = 262_144
CPU_BENCHMARK_REPEATS = 5
UDP_PROBE_PACKET_COUNT = 100
UDP_PROBE_INTERVAL_SECONDS = 30
UDP_PROBE_RESPONSE_TIMEOUT_SECONDS = 2


def _compute(message: dict[str, Any], payload: bytes) -> dict[str, Any]:
    if message.get("operation") != "sum":
        raise ValueError(f"Unsupported operation: {message.get('operation')}")
    tensor = decode_tensor(message, payload)
    if tensor.numel() == 0:
        raise ValueError("Task tensor must contain at least one element")
    return {"result": tensor.sum().item()}


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

    def send(message: dict[str, Any], payload: bytes = b"") -> None:
        with send_lock:
            send_frame(conn, message, payload)

    stream = conn.makefile("rb")
    try:
        send({
            "role": "worker",
            "node_id": node_id,
            "cpu_count": os.cpu_count() or 1,
            "device": "cpu",
            "compute_elements_per_second": compute_speed,
        })
        acknowledgement = receive_message(stream)
        if not acknowledgement or acknowledgement.get("type") != "registered":
            raise ConnectionError("Coordinator did not acknowledge registration")
        try:
            udp_probe_token = bytes.fromhex(str(acknowledgement["udp_probe_token"]))
            udp_port = int(acknowledgement["udp_port"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConnectionError("Coordinator did not provide UDP probe settings") from exc
        if len(udp_probe_token) != 16 or not 1 <= udp_port <= 65535:
            raise ConnectionError("Coordinator provided invalid UDP probe settings")
        print(f"[FSSLK] Worker ready: {node_id} (CPU, {os.cpu_count() or 1} cores)", flush=True)

        def heartbeat_loop() -> None:
            while not stop_heartbeat.wait(heartbeat_interval):
                try:
                    send({"type": "heartbeat", "node_id": node_id, "time": time.time()})
                except OSError:
                    return

        heartbeat = threading.Thread(target=heartbeat_loop, daemon=True)
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
            message_type = message.get("type")
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
