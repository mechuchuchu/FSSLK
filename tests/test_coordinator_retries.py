"""Regression tests for coordinator task retry behavior."""

from __future__ import annotations

import socket
import time
import unittest
from unittest.mock import Mock, patch

import torch

from fsslk.coordinator import (
    NETWORK_PROBE_TIMEOUT_SECONDS,
    Coordinator,
    NetworkProbe,
    WorkerNode,
)
from fsslk.protocol import receive_message, send_frame
from fsslk.tensor import decode_tensor


class CoordinatorRetryTests(unittest.TestCase):
    def make_coordinator(self, worker_count: int) -> tuple[Coordinator, list[WorkerNode]]:
        coordinator = Coordinator("127.0.0.1", 27494)
        workers = [
            WorkerNode(
                node_id=f"worker-{index}",
                conn=Mock(),
                cpu_count=1,
                device="cpu",
                control_connected=True,
            )
            for index in range(worker_count)
        ]
        coordinator.nodes = {worker.node_id: worker for worker in workers}
        return coordinator, workers

    def reply_with_sum(
        self,
        coordinator: Coordinator,
        message: dict[str, object],
        payload: bytes,
    ) -> None:
        pending = coordinator.pending[str(message["task_id"])]
        pending.future.set_result(
            {
                "type": "result",
                "ok": True,
                "result": decode_tensor(message, payload).sum().item(),
            }
        )

    def test_disconnected_worker_chunk_is_reassigned(self) -> None:
        coordinator, workers = self.make_coordinator(2)
        first, second = workers

        first.send = Mock(
            side_effect=lambda _message, _payload: coordinator._unregister_worker(first)
        )
        second.send = Mock(
            side_effect=lambda message, payload: self.reply_with_sum(
                coordinator, message, payload
            )
        )

        result, completed_workers = coordinator._distributed_sum(
            torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        )

        self.assertEqual(result, 6)
        self.assertEqual(completed_workers, [second.node_id])
        self.assertEqual(second.send.call_count, 2)
        self.assertEqual(coordinator.pending, {})

    def test_timed_out_attempt_is_removed_and_late_result_is_ignored(self) -> None:
        coordinator, workers = self.make_coordinator(2)
        first, second = workers
        first.send = Mock()
        second.send = Mock(
            side_effect=lambda message, payload: self.reply_with_sum(
                coordinator, message, payload
            )
        )

        with patch("fsslk.coordinator.TASK_TIMEOUT_SECONDS", 0.01):
            result, completed_workers = coordinator._distributed_sum(
                torch.tensor([10], dtype=torch.int64)
            )

        old_task_id = first.send.call_args.args[0]["task_id"]
        self.assertEqual(result, 10)
        self.assertEqual(completed_workers, [second.node_id])
        self.assertNotIn(old_task_id, coordinator.pending)
        self.assertEqual(coordinator.pending, {})

        late_response = {
            "type": "result",
            "task_id": old_task_id,
            "ok": True,
            "result": -100,
        }
        sender, receiver = socket.socketpair()
        stream = receiver.makefile("rb")
        try:
            send_frame(sender, late_response)
            sender.shutdown(socket.SHUT_WR)
            coordinator._worker_loop(first, stream)
        finally:
            stream.close()
            sender.close()
            receiver.close()
        self.assertEqual(coordinator.pending, {})

    def test_retries_stop_after_three_attempts(self) -> None:
        coordinator, workers = self.make_coordinator(4)
        for worker in workers:
            worker.send = Mock()

        with patch("fsslk.coordinator.TASK_TIMEOUT_SECONDS", 0.001):
            with self.assertRaisesRegex(RuntimeError, "failed after 3 attempts"):
                coordinator._distributed_sum(torch.tensor([5], dtype=torch.int64))

        self.assertEqual([worker.send.call_count for worker in workers], [1, 1, 1, 0])
        self.assertEqual(coordinator.pending, {})

    def test_network_probe_results_appear_in_status(self) -> None:
        coordinator, workers = self.make_coordinator(1)
        worker = workers[0]
        now = time.monotonic()
        worker.pending_probes["rtt-1"] = NetworkProbe("rtt", now - 0.02)
        coordinator._record_network_probe(
            worker,
            {"type": "probe_result", "probe_id": "rtt-1"},
        )
        worker.pending_probes["rtt-2"] = NetworkProbe("rtt", time.monotonic() - 0.03)
        coordinator._record_network_probe(
            worker,
            {"type": "probe_result", "probe_id": "rtt-2"},
        )
        worker.pending_probes["bandwidth-1"] = NetworkProbe(
            "bandwidth",
            time.monotonic() - 0.1,
            payload_bytes=1024,
        )
        coordinator._record_network_probe(
            worker,
            {
                "type": "bandwidth_result",
                "probe_id": "bandwidth-1",
                "received_bytes": 1024,
            },
        )
        coordinator._record_packet_loss_probe(
            worker,
            {
                "sent_packets": 20,
                "received_packets": 18,
            },
        )
        worker.pending_probes["rtt-timeout"] = NetworkProbe(
            "rtt",
            time.monotonic() - NETWORK_PROBE_TIMEOUT_SECONDS - 1,
        )
        coordinator._expire_network_probes(worker, time.monotonic())

        sender, receiver = socket.socketpair()
        stream = receiver.makefile("rb")
        try:
            coordinator._handle_client(sender, {"command": "status"})
            response = receive_message(stream)
        finally:
            stream.close()
            sender.close()
            receiver.close()

        self.assertIsNotNone(response)
        assert response is not None
        node_status = response["nodes"][0]
        self.assertEqual(node_status["state"], "healthy")
        self.assertTrue(node_status["control_connected"])
        self.assertGreater(node_status["rtt_ms"], 0)
        self.assertGreater(node_status["jitter_ms"], 0)
        self.assertGreater(node_status["bandwidth_mbps"], 0)
        self.assertEqual(node_status["packet_loss_pct"], 10.0)
        self.assertTrue(node_status["udp_probe_reachable"])
        self.assertEqual(node_status["connection_stability_pct"], 75.0)

    def test_unreachable_udp_probe_does_not_claim_packet_loss(self) -> None:
        coordinator, workers = self.make_coordinator(1)
        worker = workers[0]
        worker.packet_loss_pct = 12.5
        worker.packet_loss_samples_pct.append(12.5)

        coordinator._record_packet_loss_probe(
            worker,
            {"sent_packets": 20, "received_packets": 0},
        )

        self.assertIsNone(worker.packet_loss_pct)
        self.assertFalse(worker.udp_probe_reachable)

    def test_distributed_sum_chunk_sizes_follow_worker_metrics(self) -> None:
        coordinator, workers = self.make_coordinator(3)
        workers[0].compute_elements_per_second = 1_000_000
        workers[0].bandwidth_mbps = 1000
        workers[0].packet_loss_pct = 0
        workers[1].compute_elements_per_second = 1_000_000
        workers[1].bandwidth_mbps = 10
        workers[1].packet_loss_pct = 0
        workers[2].compute_elements_per_second = 1_000_000
        workers[2].bandwidth_mbps = 1000
        workers[2].packet_loss_pct = 50
        assigned_elements: dict[str, int] = {}

        for worker in workers:
            def send(message: dict[str, object], payload: bytes, node: WorkerNode = worker) -> None:
                assigned_elements[node.node_id] = decode_tensor(message, payload).numel()
                self.reply_with_sum(coordinator, message, payload)

            worker.send = Mock(side_effect=send)

        result, _ = coordinator._distributed_sum(torch.arange(1000, dtype=torch.int64))

        self.assertEqual(result, 499_500)
        self.assertEqual(sum(assigned_elements.values()), 1000)
        self.assertGreater(assigned_elements[workers[0].node_id], assigned_elements[workers[2].node_id])
        self.assertGreater(assigned_elements[workers[2].node_id], assigned_elements[workers[1].node_id])


if __name__ == "__main__":
    unittest.main()
