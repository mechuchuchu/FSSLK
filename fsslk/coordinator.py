"""TCP coordinator for the initial CPU-only FSSLK prototype."""

from __future__ import annotations

import math
import secrets
import socket
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass, field
from typing import Any

import torch

from .protocol import receive_frame, send_frame, send_message
from .scheduler import NodeCapacity, allocate_chunks
from .tensor import decode_tensor, encode_tensor
from .udp_probe import UDP_PROBE_PACKET, decode_probe_packet


TASK_TIMEOUT_SECONDS = 30
MAX_TASK_ATTEMPTS = 3
NETWORK_PROBE_INTERVAL_SECONDS = 5
BANDWIDTH_PROBE_INTERVAL_SECONDS = 30
NETWORK_PROBE_TIMEOUT_SECONDS = 20
BANDWIDTH_PROBE_BYTES = 256 * 1024


@dataclass
class NetworkProbe:
    kind: str
    started_at: float
    payload_bytes: int = 0


@dataclass
class WorkerNode:
    node_id: str
    conn: socket.socket
    cpu_count: int
    device: str
    compute_elements_per_second: float = 0.0
    udp_probe_token: bytes = field(default_factory=lambda: secrets.token_bytes(16))
    send_lock: threading.Lock = field(default_factory=threading.Lock)
    last_seen: float = field(default_factory=time.monotonic)
    connected: bool = True
    metrics_lock: threading.Lock = field(default_factory=threading.Lock)
    pending_probes: dict[str, NetworkProbe] = field(default_factory=dict)
    rtt_samples_ms: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    probe_outcomes: deque[bool] = field(default_factory=lambda: deque(maxlen=20))
    rtt_ms: float | None = None
    jitter_ms: float | None = None
    bandwidth_mbps: float | None = None
    packet_loss_pct: float | None = None
    packet_loss_samples_pct: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    udp_probe_reachable: bool | None = None
    last_probe_sent_at: float = 0.0
    last_bandwidth_probe_sent_at: float = 0.0

    def send(self, message: dict[str, Any], payload: bytes = b"") -> None:
        with self.send_lock:
            send_frame(self.conn, message, payload)


class Coordinator:
    def __init__(
        self,
        host: str,
        port: int,
        heartbeat_timeout: float = 15.0,
        udp_port: int | None = None,
    ):
        self.host = host
        self.port = port
        self.udp_port = udp_port if udp_port is not None else port + 1
        if not 0 <= self.udp_port <= 65535:
            raise ValueError("UDP port must be between 0 and 65535")
        self.heartbeat_timeout = heartbeat_timeout
        self.nodes: dict[str, WorkerNode] = {}
        self.nodes_lock = threading.RLock()
        self.pending: dict[str, PendingTask] = {}
        self.pending_lock = threading.Lock()
        self.server: socket.socket | None = None
        self.udp_server: socket.socket | None = None
        self.bandwidth_probe_payload = bytes(BANDWIDTH_PROBE_BYTES)

    def serve_forever(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen()
        self.server = server
        udp_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            udp_server.bind((self.host, self.udp_port))
            self.udp_port = udp_server.getsockname()[1]
            self.udp_server = udp_server
            print(
                f"[FSSLK] Coordinator listening on TCP {self.host}:{self.port} "
                f"and UDP {self.host}:{self.udp_port}",
                flush=True,
            )
            threading.Thread(target=self._network_monitor_loop, daemon=True).start()
            threading.Thread(
                target=self._udp_echo_loop,
                args=(udp_server,),
                daemon=True,
            ).start()

            while True:
                conn, address = server.accept()
                threading.Thread(
                    target=self._handle_connection,
                    args=(conn, address),
                    daemon=True,
                ).start()
        finally:
            server.close()
            udp_server.close()

    def _handle_connection(self, conn: socket.socket, address: Any) -> None:
        stream = conn.makefile("rb")
        node: WorkerNode | None = None
        try:
            frame = receive_frame(stream)
            if frame is None:
                return
            hello, payload = frame
            if hello.get("role") == "worker":
                if payload:
                    raise ValueError("worker registration must not contain a binary payload")
                node = self._register_worker(conn, hello)
                send_message(
                    conn,
                    {
                        "type": "registered",
                        "node_id": node.node_id,
                        "udp_probe_token": node.udp_probe_token.hex(),
                        "udp_port": self.udp_port,
                    },
                )
                print(f"[FSSLK] Worker joined: {node.node_id} from {address}", flush=True)
                self._worker_loop(node, stream)
            elif hello.get("role") == "client":
                self._handle_client(conn, hello, payload)
            else:
                send_message(conn, {"type": "response", "ok": False, "error": "Unknown role"})
        except (ConnectionError, OSError, ValueError) as exc:
            if node is not None:
                print(f"[FSSLK] Worker connection {node.node_id} ended: {exc}", flush=True)
        finally:
            if node is not None:
                self._unregister_worker(node)
            try:
                stream.close()
            except OSError:
                pass
            conn.close()

    def _register_worker(self, conn: socket.socket, hello: dict[str, Any]) -> WorkerNode:
        node_id = str(hello.get("node_id") or uuid.uuid4().hex[:8])
        try:
            compute_speed = float(hello.get("compute_elements_per_second") or 0)
        except (TypeError, ValueError):
            compute_speed = 0.0
        if not math.isfinite(compute_speed) or compute_speed < 0:
            compute_speed = 0.0
        node = WorkerNode(
            node_id=node_id,
            conn=conn,
            cpu_count=int(hello.get("cpu_count") or 1),
            device=str(hello.get("device") or "cpu"),
            compute_elements_per_second=compute_speed,
        )
        with self.nodes_lock:
            previous = self.nodes.get(node_id)
            if previous is not None:
                previous.connected = False
                try:
                    previous.conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                previous.conn.close()
            self.nodes[node_id] = node
        return node

    def _worker_loop(self, node: WorkerNode, stream: Any) -> None:
        while True:
            frame = receive_frame(stream)
            if frame is None:
                return
            message, payload = frame
            kind = message.get("type")
            if kind == "heartbeat":
                if payload:
                    raise ValueError("heartbeat must not contain a binary payload")
                node.last_seen = time.monotonic()
            elif kind == "result":
                if payload:
                    raise ValueError("worker results must not contain a binary payload")
                node.last_seen = time.monotonic()
                task_id = str(message.get("task_id", ""))
                with self.pending_lock:
                    pending = self.pending.get(task_id)
                    if pending is not None and pending.node is node and not pending.future.done():
                        pending.future.set_result(message)
            elif kind in {"probe_result", "bandwidth_result"}:
                if payload:
                    raise ValueError("network probe results must not contain a binary payload")
                self._record_network_probe(node, message)
            elif kind == "packet_loss_result":
                if payload:
                    raise ValueError("packet-loss results must not contain a binary payload")
                self._record_packet_loss_probe(node, message)

    def _udp_echo_loop(self, udp_server: socket.socket) -> None:
        while True:
            try:
                packet, address = udp_server.recvfrom(UDP_PROBE_PACKET.size)
            except OSError:
                return
            decoded = decode_probe_packet(packet)
            if decoded is None:
                continue
            token, _, _ = decoded
            with self.nodes_lock:
                authorized = any(
                    node.connected and secrets.compare_digest(node.udp_probe_token, token)
                    for node in self.nodes.values()
                )
            if authorized:
                try:
                    udp_server.sendto(packet, address)
                except OSError:
                    continue

    def _record_packet_loss_probe(self, node: WorkerNode, message: dict[str, Any]) -> None:
        sent = message.get("sent_packets")
        received = message.get("received_packets")
        if (
            not isinstance(sent, int)
            or isinstance(sent, bool)
            or not isinstance(received, int)
            or isinstance(received, bool)
            or sent <= 0
            or not 0 <= received <= sent
        ):
            return
        if received == 0:
            with node.metrics_lock:
                node.packet_loss_pct = None
                node.packet_loss_samples_pct.clear()
                node.udp_probe_reachable = False
                node.last_seen = time.monotonic()
            return
        loss_pct = 100 * (sent - received) / sent
        with node.metrics_lock:
            node.packet_loss_samples_pct.append(loss_pct)
            node.packet_loss_pct = sum(node.packet_loss_samples_pct) / len(
                node.packet_loss_samples_pct
            )
            node.udp_probe_reachable = True
            node.last_seen = time.monotonic()

    def _network_monitor_loop(self) -> None:
        while True:
            now = time.monotonic()
            for node in self._active_workers():
                self._expire_network_probes(node, now)
                with node.metrics_lock:
                    send_rtt_probe = (
                        now - node.last_probe_sent_at >= NETWORK_PROBE_INTERVAL_SECONDS
                    )
                    send_bandwidth_probe = (
                        now - node.last_bandwidth_probe_sent_at
                        >= BANDWIDTH_PROBE_INTERVAL_SECONDS
                    )
                    if send_rtt_probe:
                        node.last_probe_sent_at = now
                    if send_bandwidth_probe:
                        node.last_bandwidth_probe_sent_at = now

                if send_rtt_probe:
                    self._send_network_probe(node, "rtt")
                if send_bandwidth_probe:
                    self._send_network_probe(
                        node,
                        "bandwidth",
                        self.bandwidth_probe_payload,
                    )
            time.sleep(1)

    def _send_network_probe(
        self,
        node: WorkerNode,
        kind: str,
        payload: bytes = b"",
    ) -> None:
        probe_id = uuid.uuid4().hex
        probe = NetworkProbe(kind=kind, started_at=time.monotonic(), payload_bytes=len(payload))
        with node.metrics_lock:
            node.pending_probes[probe_id] = probe

        message_type = "probe" if kind == "rtt" else "bandwidth_probe"
        try:
            node.send({"type": message_type, "probe_id": probe_id}, payload)
        except OSError:
            with node.metrics_lock:
                if node.pending_probes.pop(probe_id, None) is not None:
                    node.probe_outcomes.append(False)

    def _expire_network_probes(self, node: WorkerNode, now: float) -> None:
        with node.metrics_lock:
            expired = [
                probe_id
                for probe_id, probe in node.pending_probes.items()
                if now - probe.started_at >= NETWORK_PROBE_TIMEOUT_SECONDS
            ]
            for probe_id in expired:
                del node.pending_probes[probe_id]
                node.probe_outcomes.append(False)

    def _record_network_probe(self, node: WorkerNode, message: dict[str, Any]) -> None:
        probe_id = str(message.get("probe_id", ""))
        response_type = str(message.get("type", ""))
        now = time.monotonic()
        with node.metrics_lock:
            probe = node.pending_probes.pop(probe_id, None)
            if probe is None:
                return

            expected_type = "probe_result" if probe.kind == "rtt" else "bandwidth_result"
            if response_type != expected_type:
                node.probe_outcomes.append(False)
                return
            if probe.kind == "bandwidth" and message.get("received_bytes") != probe.payload_bytes:
                node.probe_outcomes.append(False)
                return

            node.last_seen = now
            node.probe_outcomes.append(True)
            elapsed_seconds = max(now - probe.started_at, 0.000001)
            if probe.kind == "rtt":
                node.rtt_samples_ms.append(elapsed_seconds * 1000)
                samples = list(node.rtt_samples_ms)
                node.rtt_ms = sum(samples) / len(samples)
                if len(samples) > 1:
                    differences = [
                        abs(current - previous)
                        for previous, current in zip(samples, samples[1:])
                    ]
                    node.jitter_ms = sum(differences) / len(differences)
            else:
                measured_mbps = probe.payload_bytes * 8 / elapsed_seconds / 1_000_000
                if node.bandwidth_mbps is None:
                    node.bandwidth_mbps = measured_mbps
                else:
                    node.bandwidth_mbps = node.bandwidth_mbps * 0.75 + measured_mbps * 0.25

    def _unregister_worker(self, node: WorkerNode) -> None:
        with self.nodes_lock:
            node.connected = False
            if self.nodes.get(node.node_id) is node:
                del self.nodes[node.node_id]
        with self.pending_lock:
            for pending in self.pending.values():
                if pending.node is node and not pending.future.done():
                    pending.future.set_exception(
                        ConnectionError(f"Worker {node.node_id} disconnected")
                    )
        with node.metrics_lock:
            node.probe_outcomes.extend(False for _ in node.pending_probes)
            node.pending_probes.clear()
        print(f"[FSSLK] Worker left: {node.node_id}", flush=True)

    def _active_workers(self) -> list[WorkerNode]:
        now = time.monotonic()
        with self.nodes_lock:
            return [
                node
                for node in self.nodes.values()
                if node.connected and now - node.last_seen <= self.heartbeat_timeout
            ]

    def _handle_client(
        self,
        conn: socket.socket,
        request: dict[str, Any],
        payload: bytes = b"",
    ) -> None:
        command = request.get("command")
        if command == "status":
            if payload:
                send_message(conn, {"type": "response", "ok": False, "error": "status does not accept a payload"})
                return
            now = time.monotonic()
            with self.nodes_lock:
                registered_nodes = [node for node in self.nodes.values() if node.connected]

            nodes = []
            for node in registered_nodes:
                with node.metrics_lock:
                    stability = (
                        100 * sum(node.probe_outcomes) / len(node.probe_outcomes)
                        if node.probe_outcomes
                        else None
                    )
                    nodes.append(
                        {
                            "node_id": node.node_id,
                            "device": node.device,
                            "cpu_count": node.cpu_count,
                            "compute_elements_per_second": round(
                                node.compute_elements_per_second, 1
                            ),
                            "state": "active" if now - node.last_seen <= self.heartbeat_timeout else "stale",
                            "heartbeat_age_seconds": round(now - node.last_seen, 1),
                            "rtt_ms": round(node.rtt_ms, 2) if node.rtt_ms is not None else None,
                            "jitter_ms": round(node.jitter_ms, 2) if node.jitter_ms is not None else None,
                            "bandwidth_mbps": (
                                round(node.bandwidth_mbps, 2)
                                if node.bandwidth_mbps is not None
                                else None
                            ),
                            "packet_loss_pct": (
                                round(node.packet_loss_pct, 2)
                                if node.packet_loss_pct is not None
                                else None
                            ),
                            "udp_probe_reachable": node.udp_probe_reachable,
                            "connection_stability_pct": (
                                round(stability, 1) if stability is not None else None
                            ),
                        }
                    )
            send_message(conn, {"type": "response", "ok": True, "nodes": nodes})
            return

        if command == "sum":
            try:
                values = decode_tensor(request, payload)
                if values.numel() == 0:
                    raise ValueError("tensor must contain at least one element")
                result, workers = self._distributed_sum(values)
                send_message(
                    conn,
                    {"type": "response", "ok": True, "result": result, "workers": workers},
                )
            except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
                send_message(conn, {"type": "response", "ok": False, "error": str(exc)})
            return

        send_message(conn, {"type": "response", "ok": False, "error": "Unknown command"})

    def _distributed_sum(self, values: torch.Tensor) -> tuple[int | float, list[str]]:
        if values.numel() == 0:
            raise ValueError("tensor must contain at least one element")
        workers = self._active_workers()
        if not workers:
            raise RuntimeError("No active workers are connected")

        flat_values = values.reshape(-1).contiguous()
        allocations = allocate_chunks(
            flat_values.numel(),
            [
                NodeCapacity(
                    node_id=node.node_id,
                    compute_elements_per_second=(
                        node.compute_elements_per_second or float(node.cpu_count)
                    ),
                    bandwidth_mbps=node.bandwidth_mbps,
                    packet_loss_pct=node.packet_loss_pct,
                    rtt_ms=node.rtt_ms,
                )
                for node in workers
            ],
            flat_values.element_size(),
        )
        tasks: list[SumTask] = []
        partials: list[int | float] = []
        completed_workers: list[str] = []
        try:
            offset = 0
            for node in workers:
                chunk_elements = allocations.get(node.node_id, 0)
                if chunk_elements > 0:
                    chunk = flat_values[offset : offset + chunk_elements]
                    offset += chunk_elements
                    task = SumTask(values=chunk)
                    tasks.append(task)
                    self._start_sum_attempt(task, node)

            for task in tasks:
                while True:
                    pending = task.pending
                    if pending is None:
                        raise RuntimeError("Task has no active worker attempt")
                    try:
                        response = pending.future.result(timeout=TASK_TIMEOUT_SECONDS)
                        if not response.get("ok"):
                            raise RuntimeError(
                                f"Worker {pending.node.node_id} failed: "
                                f"{response.get('error', 'unknown error')}"
                            )
                        result = response.get("result")
                        if not isinstance(result, (int, float)) or isinstance(result, bool):
                            raise RuntimeError(
                                f"Worker {pending.node.node_id} returned an invalid sum result"
                            )
                        partials.append(result)
                        if pending.node.node_id not in completed_workers:
                            completed_workers.append(pending.node.node_id)
                        self._clear_sum_attempt(task, pending)
                        break
                    except (TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
                        self._clear_sum_attempt(task, pending)
                        self._retry_sum_task(task, exc)
        finally:
            for task in tasks:
                if task.pending is not None:
                    self._clear_sum_attempt(task, task.pending)

        return sum(partials), completed_workers

    def _start_sum_attempt(self, task: SumTask, node: WorkerNode) -> None:
        task_id = uuid.uuid4().hex
        pending = PendingTask(task_id=task_id, node=node, future=Future())
        task.pending = pending
        task.attempted_node_ids.add(node.node_id)
        metadata, payload = encode_tensor(task.values)
        with self.pending_lock:
            self.pending[task_id] = pending

        try:
            node.send(
                {
                    "type": "task",
                    "task_id": task_id,
                    "operation": "sum",
                    **metadata,
                },
                payload,
            )
        except OSError as exc:
            with self.pending_lock:
                if self.pending.get(task_id) is pending and not pending.future.done():
                    pending.future.set_exception(exc)

    def _clear_sum_attempt(self, task: SumTask, pending: PendingTask) -> None:
        with self.pending_lock:
            if self.pending.get(pending.task_id) is pending:
                del self.pending[pending.task_id]
        if task.pending is pending:
            task.pending = None

    def _retry_sum_task(self, task: SumTask, last_error: Exception) -> None:
        if len(task.attempted_node_ids) >= MAX_TASK_ATTEMPTS:
            raise RuntimeError(
                f"Task failed after {len(task.attempted_node_ids)} attempts: {last_error}"
            ) from last_error

        candidates = [
            node
            for node in self._active_workers()
            if node.node_id not in task.attempted_node_ids
        ]
        if not candidates:
            raise RuntimeError(
                f"Task failed and no other active worker is available: {last_error}"
            ) from last_error

        self._start_sum_attempt(task, candidates[0])


@dataclass
class PendingTask:
    task_id: str
    node: WorkerNode
    future: Future[dict[str, Any]]


@dataclass
class SumTask:
    values: torch.Tensor
    attempted_node_ids: set[str] = field(default_factory=set)
    pending: PendingTask | None = None
