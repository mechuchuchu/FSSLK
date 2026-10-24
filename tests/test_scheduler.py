"""Tests for metric-driven deterministic task allocation."""

from __future__ import annotations

import unittest

from fsslk.scheduler import NodeCapacity, allocate_chunks


class SchedulerTests(unittest.TestCase):
    def test_allocation_sums_to_total_and_favors_faster_hardware(self) -> None:
        allocations = allocate_chunks(
            1000,
            [
                NodeCapacity("fast", 2_000_000, 1000, 0),
                NodeCapacity("slow", 1_000_000, 1000, 0),
            ],
            element_size_bytes=8,
        )

        self.assertEqual(sum(allocations.values()), 1000)
        self.assertGreater(allocations["fast"], allocations["slow"])

    def test_low_bandwidth_and_packet_loss_reduce_assigned_work(self) -> None:
        allocations = allocate_chunks(
            1000,
            [
                NodeCapacity("clean", 1_000_000, 1000, 0),
                NodeCapacity("lossy", 1_000_000, 100, 50),
            ],
            element_size_bytes=8,
        )

        self.assertGreater(allocations["clean"], allocations["lossy"])
        self.assertEqual(sum(allocations.values()), 1000)

    def test_missing_network_metrics_fall_back_to_compute_capacity(self) -> None:
        allocations = allocate_chunks(
            17,
            [NodeCapacity("a", 2_000_000), NodeCapacity("b", 1_000_000)],
            element_size_bytes=8,
        )

        self.assertEqual(sum(allocations.values()), 17)
        self.assertGreater(allocations["a"], allocations["b"])

    def test_high_latency_worker_gets_less_of_a_small_workload(self) -> None:
        allocations = allocate_chunks(
            10,
            [
                NodeCapacity("near", 1_000_000, 100, 0, 1),
                NodeCapacity("far", 1_000_000, 100, 0, 100),
            ],
            element_size_bytes=8,
        )

        self.assertGreater(allocations.get("near", 0), allocations.get("far", 0))
        self.assertEqual(sum(allocations.values()), 10)

    def test_invalid_work_and_element_size_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            allocate_chunks(0, [NodeCapacity("a", 1)], 8)
        with self.assertRaises(ValueError):
            allocate_chunks(1, [NodeCapacity("a", 1)], 0)


if __name__ == "__main__":
    unittest.main()
