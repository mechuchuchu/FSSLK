"""TCP coordinator for the initial CPU-only FSSLK prototype."""

from __future__ import annotations

import math
import socket
import threading
import time
import uuid
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass, field
from typing import Any

from .protocol import receive_message, send_message


@dataclass
class WorkerNode:
    node_id: str
    conn: socket.socket
    cpu_count: int
    device: str
    send_lock: threading.Lock = field(default_factory=threading.Lock)
    last_seen: float = field(default_factory=time.monotonic)
    connected: bool = True

    def send(self, message: dict[str, Any]) -> None:
        with self.send_lock:
            send_message(self.conn, message)


class Coordinator:
    def __init__(self, host: str, port: int, heartbeat_timeout: float = 15.0):
        self.host = host
        self.port = port
        self.heartbeat_timeout = heartbeat_timeout
        self.nodes: dict[str, WorkerNode] = {}
        self.nodes_lock = threading.RLock()
        self.pending: dict[str, Future[dict[str, Any]]] = {}
        self.pending_lock = threading.Lock()
        self.server: socket.socket | None = None

    def serve_forever(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen()
        self.server = server
        print(f"[FSSLK] Coordinator listening on {self.host}:{self.port}", flush=True)

        try:
            while True:
                conn, address = server.accept()
                threading.Thread(
                    target=self._handle_connection,
                    args=(conn, address),
                    daemon=True,
                ).start()
        finally:
            server.close()

    def _handle_connection(self, conn: socket.socket, address: Any) -> None:
        stream = conn.makefile("rb")
        node: WorkerNode | None = None
        try:
            hello = receive_message(stream)
            if hello is None:
                return
            if hello.get("role") == "worker":
                node = self._register_worker(conn, hello)
                send_message(conn, {"type": "registered", "node_id": node.node_id})
                print(f"[FSSLK] Worker joined: {node.node_id} from {address}", flush=True)
                self._worker_loop(node, stream)
            elif hello.get("role") == "client":
                self._handle_client(conn, hello)
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
        node = WorkerNode(
            node_id=node_id,
            conn=conn,
            cpu_count=int(hello.get("cpu_count") or 1),
            device=str(hello.get("device") or "cpu"),
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
            message = receive_message(stream)
            if message is None:
                return
            kind = message.get("type")
            if kind == "heartbeat":
                node.last_seen = time.monotonic()
            elif kind == "result":
                node.last_seen = time.monotonic()
                task_id = str(message.get("task_id", ""))
                with self.pending_lock:
                    future = self.pending.get(task_id)
                if future is not None and not future.done():
                    future.set_result(message)

    def _unregister_worker(self, node: WorkerNode) -> None:
        with self.nodes_lock:
            node.connected = False
            if self.nodes.get(node.node_id) is node:
                del self.nodes[node.node_id]
        print(f"[FSSLK] Worker left: {node.node_id}", flush=True)

    def _active_workers(self) -> list[WorkerNode]:
        now = time.monotonic()
        with self.nodes_lock:
            return [
                node
                for node in self.nodes.values()
                if node.connected and now - node.last_seen <= self.heartbeat_timeout
            ]

    def _handle_client(self, conn: socket.socket, request: dict[str, Any]) -> None:
        command = request.get("command")
        if command == "status":
            now = time.monotonic()
            with self.nodes_lock:
                nodes = [
                    {
                        "node_id": node.node_id,
                        "device": node.device,
                        "cpu_count": node.cpu_count,
                        "state": "active" if now - node.last_seen <= self.heartbeat_timeout else "stale",
                        "heartbeat_age_seconds": round(now - node.last_seen, 1),
                    }
                    for node in self.nodes.values()
                    if node.connected
                ]
            send_message(conn, {"type": "response", "ok": True, "nodes": nodes})
            return

        if command == "sum":
            try:
                values = request.get("values")
                if not isinstance(values, list) or not values:
                    raise ValueError("values must be a non-empty list")
                if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
                    raise ValueError("values must contain only numbers")
                result, workers = self._distributed_sum(values)
                send_message(
                    conn,
                    {"type": "response", "ok": True, "result": result, "workers": workers},
                )
            except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
                send_message(conn, {"type": "response", "ok": False, "error": str(exc)})
            return

        send_message(conn, {"type": "response", "ok": False, "error": "Unknown command"})

    def _distributed_sum(self, values: list[int | float]) -> tuple[int | float, list[str]]:
        workers = self._active_workers()
        if not workers:
            raise RuntimeError("No active workers are connected")

        worker_count = min(len(workers), len(values))
        selected = workers[:worker_count]
        chunk_size = math.ceil(len(values) / worker_count)
        assignments: list[tuple[WorkerNode, str, list[int | float]]] = []
        for index, node in enumerate(selected):
            chunk = values[index * chunk_size : (index + 1) * chunk_size]
            if chunk:
                assignments.append((node, uuid.uuid4().hex, chunk))

        futures: list[tuple[WorkerNode, Future[dict[str, Any]]]] = []
        try:
            with self.pending_lock:
                for node, task_id, _ in assignments:
                    self.pending[task_id] = Future()
                    futures.append((node, self.pending[task_id]))

            for node, task_id, chunk in assignments:
                node.send({"type": "task", "task_id": task_id, "operation": "sum", "values": chunk})

            partials: list[int | float] = []
            for node, future in futures:
                response = future.result(timeout=30)
                if not response.get("ok"):
                    raise RuntimeError(f"Worker {node.node_id} failed: {response.get('error', 'unknown error')}")
                partials.append(response["result"])
            total = sum(partials)
            return total, [node.node_id for node, _ in futures]
        except TimeoutError as exc:
            raise TimeoutError("A worker did not finish the task within 30 seconds") from exc
        except OSError as exc:
            raise RuntimeError(f"Could not send work to a worker: {exc}") from exc
        finally:
            with self.pending_lock:
                for _, task_id, _ in assignments:
                    self.pending.pop(task_id, None)
