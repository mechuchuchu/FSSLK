"""Tests for the worker's separate control connection and liveness states."""

from __future__ import annotations

from concurrent.futures import Future
import socket
import threading
import time
import unittest
from unittest.mock import Mock

from fsslk.coordinator import Coordinator, PendingTask, WorkerNode
from fsslk.protocol import receive_message, send_frame, send_message
from fsslk.worker import _control_heartbeat_loop


class CoordinatorControlTests(unittest.TestCase):
    def make_node(self, coordinator: Coordinator) -> WorkerNode:
        node = WorkerNode(
            node_id="worker-a",
            conn=Mock(),
            cpu_count=1,
            device="cpu",
            heartbeat_interval=0.05,
            control_token="a" * 32,
        )
        coordinator.nodes[node.node_id] = node
        return node

    def test_control_heartbeat_ack_and_liveness_transitions(self) -> None:
        coordinator = Coordinator("127.0.0.1", 27494, heartbeat_timeout=0.3)
        node = self.make_node(coordinator)
        server_conn, client_conn = socket.socketpair()
        client_stream = client_conn.makefile("rb")
        handler = threading.Thread(
            target=coordinator._handle_connection,
            args=(server_conn, ("local", 0)),
            daemon=True,
        )
        handler.start()

        try:
            send_frame(
                client_conn,
                {
                    "role": "worker_control",
                    "node_id": node.node_id,
                    "control_token": node.control_token,
                },
            )
            self.assertEqual(
                receive_message(client_stream),
                {"type": "control_registered", "node_id": node.node_id},
            )
            self.assertEqual(coordinator._worker_liveness(node), "healthy")

            send_frame(client_conn, {"type": "heartbeat", "sequence": 1})
            self.assertEqual(
                receive_message(client_stream),
                {"type": "heartbeat_ack", "sequence": 1},
            )
            self.assertEqual(coordinator._worker_liveness(node), "healthy")
            self.assertEqual(coordinator._active_workers(), [node])

            client_conn.shutdown(socket.SHUT_RDWR)
            handler.join(timeout=2)
            self.assertFalse(handler.is_alive())
            self.assertFalse(node.control_connected)
            self.assertEqual(coordinator._worker_liveness(node), "suspect")
            self.assertEqual(
                coordinator._worker_liveness(
                    node, node.last_heartbeat + coordinator.heartbeat_timeout + 0.01
                ),
                "unreachable",
            )
            self.assertEqual(coordinator._active_workers(), [])
        finally:
            client_stream.close()
            client_conn.close()

    def test_invalid_control_token_is_rejected(self) -> None:
        coordinator = Coordinator("127.0.0.1", 27494)
        node = self.make_node(coordinator)
        server_conn, client_conn = socket.socketpair()
        client_stream = client_conn.makefile("rb")
        handler = threading.Thread(
            target=coordinator._handle_connection,
            args=(server_conn, ("local", 0)),
            daemon=True,
        )
        handler.start()

        try:
            send_frame(
                client_conn,
                {
                    "role": "worker_control",
                    "node_id": node.node_id,
                    "control_token": "wrong-token",
                },
            )
            response = receive_message(client_stream)
            self.assertIsNotNone(response)
            assert response is not None
            self.assertEqual(response["type"], "control_rejected")
            self.assertFalse(node.control_connected)
        finally:
            client_conn.close()
            client_stream.close()
            handler.join(timeout=2)

    def test_expired_control_lease_fails_pending_worker_requests(self) -> None:
        coordinator = Coordinator("127.0.0.1", 27494, heartbeat_timeout=0.3)
        node = self.make_node(coordinator)
        node.control_connected = True
        node.last_heartbeat = time.monotonic() - 0.31
        future: Future = Future()
        coordinator.pending["rpc-1"] = PendingTask("rpc-1", node, future)

        coordinator._expire_unreachable_workers()

        with self.assertRaisesRegex(ConnectionError, "heartbeat lease expired"):
            future.result()


class WorkerControlHeartbeatTests(unittest.TestCase):
    def test_worker_uses_distinct_connection_and_acknowledged_sequences(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        host, port = listener.getsockname()
        stop = threading.Event()
        heartbeat_received = threading.Event()
        errors: list[BaseException] = []

        def serve_control() -> None:
            try:
                conn, _address = listener.accept()
                with conn:
                    stream = conn.makefile("rb")
                    try:
                        hello = receive_message(stream)
                        self.assertEqual(hello["role"], "worker_control")
                        self.assertEqual(hello["node_id"], "worker-a")
                        self.assertEqual(hello["control_token"], "control-token")
                        send_message(
                            conn,
                            {"type": "control_registered", "node_id": "worker-a"},
                        )
                        heartbeat = receive_message(stream)
                        self.assertEqual(
                            heartbeat,
                            {"type": "heartbeat", "sequence": 1},
                        )
                        send_message(conn, {"type": "heartbeat_ack", "sequence": 1})
                        heartbeat_received.set()
                        while receive_message(stream) is not None:
                            pass
                    finally:
                        stream.close()
            except BaseException as exc:
                errors.append(exc)

        server = threading.Thread(target=serve_control, daemon=True)
        server.start()
        worker = threading.Thread(
            target=_control_heartbeat_loop,
            args=(host, port, "worker-a", "control-token", 0.05, stop),
            daemon=True,
        )
        worker.start()
        try:
            self.assertTrue(heartbeat_received.wait(2))
        finally:
            stop.set()
            worker.join(timeout=2)
            listener.close()
            server.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertFalse(server.is_alive())
        self.assertEqual(errors, [])

    def test_worker_reconnects_after_control_channel_closes(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(2)
        host, port = listener.getsockname()
        stop = threading.Event()
        reconnected = threading.Event()
        errors: list[BaseException] = []

        def serve_control() -> None:
            try:
                for attempt in range(2):
                    conn, _address = listener.accept()
                    with conn:
                        stream = conn.makefile("rb")
                        try:
                            hello = receive_message(stream)
                            self.assertEqual(hello["role"], "worker_control")
                            send_message(
                                conn,
                                {"type": "control_registered", "node_id": "worker-a"},
                            )
                            heartbeat = receive_message(stream)
                            self.assertEqual(heartbeat["sequence"], 1)
                            if attempt == 0:
                                continue  # Drop the first channel before acknowledging it.
                            send_message(
                                conn,
                                {"type": "heartbeat_ack", "sequence": 1},
                            )
                            reconnected.set()
                            while receive_message(stream) is not None:
                                pass
                        finally:
                            stream.close()
            except BaseException as exc:
                errors.append(exc)

        server = threading.Thread(target=serve_control, daemon=True)
        server.start()
        worker = threading.Thread(
            target=_control_heartbeat_loop,
            args=(host, port, "worker-a", "control-token", 0.05, stop),
            daemon=True,
        )
        worker.start()
        try:
            self.assertTrue(reconnected.wait(2))
        finally:
            stop.set()
            worker.join(timeout=2)
            server.join(timeout=2)
            listener.close()

        self.assertFalse(worker.is_alive())
        self.assertFalse(server.is_alive())
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
