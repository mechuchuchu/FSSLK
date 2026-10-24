"""Tests for coordinator routing of Worker stage commands."""

from __future__ import annotations

from concurrent.futures import Future
import socket
import unittest
from unittest.mock import Mock, patch

from fsslk.coordinator import Coordinator, PendingTask, WorkerNode
from fsslk.protocol import send_frame


class CoordinatorStageRelayTests(unittest.TestCase):
    def make_coordinator(self) -> tuple[Coordinator, WorkerNode]:
        coordinator = Coordinator("127.0.0.1", 27494)
        worker = WorkerNode(
            node_id="worker-a",
            conn=Mock(),
            cpu_count=1,
            device="cpu",
            control_connected=True,
        )

        def reply_to_stage(message: dict, payload: bytes) -> None:
            self.assertEqual(payload, b"input-payload")
            pending = coordinator.pending[message["task_id"]]
            pending.future.set_result(
                (
                    {
                        "type": "stage_result",
                        "task_id": message["task_id"],
                        "ok": True,
                        "tensors": [{"name": "y"}],
                    },
                    b"output-payload",
                )
            )

        worker.send = Mock(side_effect=reply_to_stage)
        coordinator.nodes = {worker.node_id: worker}
        return coordinator, worker

    def test_stage_request_is_routed_and_binary_reply_is_relayed(self) -> None:
        coordinator, worker = self.make_coordinator()
        conn = object()
        request = {
            "worker": "worker-a",
            "stage_id": "stage-a",
            "tensors": [{"name": "x"}],
            "untrusted_extra": "must not be forwarded",
        }

        with patch("fsslk.coordinator.send_frame") as send_frame:
            coordinator._relay_stage(conn, "stage_run", request, b"input-payload")

        forwarded = worker.send.call_args.args[0]
        self.assertEqual(forwarded["type"], "stage")
        self.assertEqual(forwarded["op"], "run")
        self.assertEqual(forwarded["stage_id"], "stage-a")
        self.assertEqual(forwarded["tensors"], request["tensors"])
        self.assertNotIn("untrusted_extra", forwarded)
        self.assertEqual(worker.send.call_args.args[1], b"input-payload")

        response = send_frame.call_args.args[1]
        self.assertEqual(response["type"], "response")
        self.assertTrue(response["ok"])
        self.assertEqual(response["tensors"], [{"name": "y"}])
        self.assertEqual(send_frame.call_args.args[2], b"output-payload")
        self.assertEqual(coordinator.pending, {})

    def test_unknown_worker_returns_error_without_creating_pending_task(self) -> None:
        coordinator = Coordinator("127.0.0.1", 27494)

        with patch("fsslk.coordinator.send_message") as send_message:
            coordinator._relay_stage(
                object(), "stage_run", {"worker": "missing"}, b""
            )

        response = send_message.call_args.args[1]
        self.assertFalse(response["ok"])
        self.assertIn("Unknown or disconnected worker", response["error"])
        self.assertEqual(coordinator.pending, {})

    def test_worker_loop_delivers_stage_reply_with_its_binary_payload(self) -> None:
        coordinator = Coordinator("127.0.0.1", 27494)
        worker = WorkerNode(
            node_id="worker-a", conn=Mock(), cpu_count=1, device="cpu"
        )
        task_id = "stage-task"
        future: Future = Future()
        coordinator.pending[task_id] = PendingTask(task_id, worker, future)
        sender, receiver = socket.socketpair()
        stream = receiver.makefile("rb")
        reply = {
            "type": "stage_result",
            "task_id": task_id,
            "ok": True,
            "tensors": [{"name": "y"}],
        }
        try:
            send_frame(sender, reply, b"activation-bytes")
            sender.shutdown(socket.SHUT_WR)
            coordinator._worker_loop(worker, stream)
        finally:
            stream.close()
            sender.close()
            receiver.close()

        self.assertTrue(future.done())
        message, payload = future.result()
        self.assertEqual(message, reply)
        self.assertEqual(payload, b"activation-bytes")


if __name__ == "__main__":
    unittest.main()
