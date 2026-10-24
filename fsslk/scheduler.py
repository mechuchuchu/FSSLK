"""Simple deterministic chunk allocation based on measured node capacity."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NodeCapacity:
    node_id: str
    compute_elements_per_second: float
    bandwidth_mbps: float | None = None
    packet_loss_pct: float | None = None
    rtt_ms: float | None = None


def allocate_chunks(
    total_elements: int,
    nodes: list[NodeCapacity],
    element_size_bytes: int,
) -> dict[str, int]:
    """Split work in proportion to combined compute and transfer capacity.

    Compute and network throughput are treated as serial costs. The loss
    measurement reduces effective network throughput linearly as a first-pass
    TCP penalty model.
    """
    if total_elements <= 0:
        raise ValueError("total_elements must be positive")
    if element_size_bytes <= 0:
        raise ValueError("element_size_bytes must be positive")
    if not nodes:
        return {}

    capacities = {
        node.node_id: _positive(node.compute_elements_per_second, 1.0)
        for node in nodes
    }
    known_network_rates = [
        node.bandwidth_mbps * 1_000_000 / (8 * element_size_bytes)
        for node in nodes
        if node.bandwidth_mbps is not None
        and math.isfinite(node.bandwidth_mbps)
        and node.bandwidth_mbps > 0
    ]
    fallback_network_rate = max(known_network_rates, default=max(capacities.values()))

    seconds_per_element: dict[str, float] = {}
    latency_seconds: dict[str, float] = {}
    for node in nodes:
        compute_rate = capacities[node.node_id]
        if (
            node.bandwidth_mbps is not None
            and math.isfinite(node.bandwidth_mbps)
            and node.bandwidth_mbps > 0
        ):
            network_rate = node.bandwidth_mbps * 1_000_000 / (8 * element_size_bytes)
        else:
            network_rate = fallback_network_rate

        loss = node.packet_loss_pct
        loss_factor = 1.0
        if loss is not None and math.isfinite(loss):
            loss_factor = max(0.01, 1.0 - min(max(loss, 0.0), 100.0) / 100.0)
        effective_network_rate = max(network_rate * loss_factor, 1.0)
        seconds_per_element[node.node_id] = (
            1.0 / compute_rate + 1.0 / effective_network_rate
        )
        rtt = node.rtt_ms
        latency_seconds[node.node_id] = (
            max(rtt, 0.0) / 1000
            if rtt is not None and math.isfinite(rtt)
            else 0.0
        )

    # Find the completion time where all nodes' predicted chunk sizes sum to
    # the workload. Each node pays one RTT plus its per-element transfer and
    # compute cost.
    low = min(latency_seconds.values())
    high = max(latency_seconds.values()) + total_elements * max(
        seconds_per_element.values()
    )
    for _ in range(80):
        target_time = (low + high) / 2
        assigned = sum(
            max(0.0, (target_time - latency_seconds[node_id]) / cost)
            for node_id, cost in seconds_per_element.items()
        )
        if assigned < total_elements:
            low = target_time
        else:
            high = target_time

    raw_allocations = [
        max(0.0, (high - latency_seconds[node.node_id]) / seconds_per_element[node.node_id])
        for node in nodes
    ]
    raw_total = sum(raw_allocations)
    exact_allocations = [
        allocation * total_elements / raw_total for allocation in raw_allocations
    ]
    integer_allocations = [int(allocation) for allocation in exact_allocations]
    remaining = total_elements - sum(integer_allocations)
    remainder_order = sorted(
        range(len(nodes)),
        key=lambda index: exact_allocations[index] - integer_allocations[index],
        reverse=True,
    )
    for index in remainder_order[:remaining]:
        integer_allocations[index] += 1

    return {
        node.node_id: count
        for node, count in zip(nodes, integer_allocations)
        if count > 0
    }


def _positive(value: float, fallback: float) -> float:
    if not math.isfinite(value) or value <= 0:
        return fallback
    return value
