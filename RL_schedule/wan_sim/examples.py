"""Explicitly synthetic topology for quick API exploration.

The rates here are placeholders. Research runs should supply measured node and
WAN-link specifications instead of treating these numbers as a benchmark.
"""

from __future__ import annotations

from .types import LinkSpec, NodeSpec, Topology, Workload


def four_region_topology() -> Topology:
    regions = ("R1", "R2", "R3", "R4")
    node_rates = (1.2e12, 1.0e12, 0.9e12, 0.8e12, 1.1e12, 0.95e12, 0.85e12, 0.75e12)
    nodes = tuple(
        NodeSpec(
            node_id=f"node-{index + 1}",
            region=regions[index // 2],
            flops_per_second=node_rates[index],
            memory_bytes=80 * 1024**3,
            max_local_batch=1024,
        )
        for index in range(8)
    )
    links = [
        LinkSpec(
            link_id=f"local-{region}",
            region_a=region,
            region_b=region,
            bytes_per_second=100e9,
            latency_seconds=0.0002,
        )
        for region in regions
    ]
    for index, region in enumerate(regions):
        next_region = regions[(index + 1) % len(regions)]
        links.append(
            LinkSpec(
                link_id=f"wan-{region}-{next_region}",
                region_a=region,
                region_b=next_region,
                bytes_per_second=5e9,
                latency_seconds=0.025,
            )
        )
    topology = Topology(nodes=nodes, links=tuple(links))
    topology.validate()
    return topology


def example_workload() -> Workload:
    return Workload(
        global_batch_size=768,
        target_samples=3072,
        flops_per_sample=1e10,
        gradient_bytes=100 * 1024**2,
        default_micro_batch_size=32,
        model_state_bytes=4 * 1024**3,
        activation_bytes_per_sample=16 * 1024**2,
        # Illustrative only; calibrate this using the target model and node.
        batch_launch_overhead_seconds=0.00025,
    )
