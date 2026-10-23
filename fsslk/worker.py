"""CPU worker that receives and computes small PyTorch tensor tasks."""

from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from typing import Any

import torch

from .protocol import receive_message, send_message


def _compute(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("operation") != "sum":
        raise ValueError(f"Unsupported operation: {message.get('operation')}")
    values = message.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError("Task values must be a non-empty list")
    tensor = torch.tensor(values, device="cpu")
    return {"result": tensor.sum().item()}


def run_worker(host: str, port: int, node_id: str | None = None, heartbeat_interval: float = 2.0) -> None:
    identity = node_id or os.environ.get("FSSLK_NODE_ID") or socket.gethostname() or uuid.uuid4().hex[:8]
    delay = 1.0
    while True:
        try:
            _run_connection(host, port, identity, heartbeat_interval)
            delay = 1.0
        except (ConnectionError, OSError, ValueError) as exc:
            print(f"[FSSLK] {identity}: coordinator unavailable ({exc}); retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 10.0)


def _run_connection(host: str, port: int, node_id: str, heartbeat_interval: float) -> None:
    conn = socket.create_connection((host, port), timeout=10)
    conn.settimeout(None)
    send_lock = threading.Lock()
    stop_heartbeat = threading.Event()

    def send(message: dict[str, Any]) -> None:
        with send_lock:
            send_message(conn, message)

    stream = conn.makefile("rb")
    try:
        send({
            "role": "worker",
            "node_id": node_id,
            "cpu_count": os.cpu_count() or 1,
            "device": "cpu",
        })
        acknowledgement = receive_message(stream)
        if not acknowledgement or acknowledgement.get("type") != "registered":
            raise ConnectionError("Coordinator did not acknowledge registration")
        print(f"[FSSLK] Worker ready: {node_id} (CPU, {os.cpu_count() or 1} cores)", flush=True)

        def heartbeat_loop() -> None:
            while not stop_heartbeat.wait(heartbeat_interval):
                try:
                    send({"type": "heartbeat", "node_id": node_id, "time": time.time()})
                except OSError:
                    return

        heartbeat = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat.start()

        while True:
            message = receive_message(stream)
            if message is None:
                raise ConnectionError("Coordinator closed the connection")
            if message.get("type") != "task":
                continue
            task_id = str(message.get("task_id", ""))
            try:
                output = _compute(message)
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
