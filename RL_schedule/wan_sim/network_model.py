"""Fixed-route WAN and ring AllReduce timing with shared link capacities."""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from .types import LinkSpec, LinkState, NodeSpec, Topology


@dataclass(frozen=True)
class CollectiveEstimate:
    duration_seconds: float
    links_used: frozenset[str]
    bytes_transferred: int


class WANNetworkModel:
    def __init__(self, topology: Topology) -> None:
        topology.validate()
        self.topology = topology
        self.nodes = {node.node_id: node for node in topology.nodes}
        self.links = {link.link_id: link for link in topology.links}
        all_up = {link_id: LinkState.UP for link_id in self.links}
        regions = sorted({node.region for node in topology.nodes})
        self._fixed_routes = {
            (source, target): self._shortest_route(source, target, all_up)
            for source in regions
            for target in regions
        }

    def route(
        self,
        region_a: str,
        region_b: str,
        link_states: Mapping[str, LinkState],
    ) -> tuple[LinkSpec, ...] | None:
        if region_a == region_b:
            local = next(
                (
                    link
                    for link in self.topology.links
                    if link.region_a == region_a and link.region_b == region_a
                ),
                None,
            )
            path = () if local is None else (local,)
        else:
            path = self._fixed_routes.get((region_a, region_b))
        if path is None:
            return None
        if any(link_states.get(link.link_id, LinkState.UP) != LinkState.UP for link in path):
            return None
        return path

    def _shortest_route(
        self,
        region_a: str,
        region_b: str,
        link_states: Mapping[str, LinkState],
    ) -> tuple[LinkSpec, ...] | None:
        if region_a == region_b:
            local = next(
                (
                    link
                    for link in self.topology.links
                    if link.region_a == region_a and link.region_b == region_a
                ),
                None,
            )
            return () if local is None else (local,)
        adjacency: dict[str, list[tuple[str, LinkSpec]]] = {}
        for link in self.topology.links:
            if link.region_a == link.region_b:
                continue
            if link_states.get(link.link_id, LinkState.UP) != LinkState.UP:
                continue
            adjacency.setdefault(link.region_a, []).append((link.region_b, link))
            adjacency.setdefault(link.region_b, []).append((link.region_a, link))

        queue: list[tuple[float, int, str, tuple[LinkSpec, ...]]] = []
        sequence = 0
        heapq.heappush(queue, (0.0, sequence, region_a, ()))
        best = {region_a: 0.0}
        while queue:
            distance, _, region, path = heapq.heappop(queue)
            if region == region_b:
                return path
            if distance > best.get(region, float("inf")):
                continue
            for neighbor, link in adjacency.get(region, ()):
                candidate = distance + link.latency_seconds
                if candidate >= best.get(neighbor, float("inf")):
                    continue
                best[neighbor] = candidate
                sequence += 1
                heapq.heappush(queue, (candidate, sequence, neighbor, path + (link,)))
        return None

    def ring_allreduce(
        self,
        participant_ids: Sequence[str],
        gradient_bytes: int,
        link_states: Mapping[str, LinkState],
    ) -> CollectiveEstimate | None:
        participants = tuple(participant_ids)
        if not participants:
            raise ValueError("AllReduce requires at least one participant")
        if gradient_bytes < 0:
            raise ValueError("gradient_bytes must be non-negative")
        if len(participants) == 1 or gradient_bytes == 0:
            return CollectiveEstimate(0.0, frozenset(), 0)

        chunk_bytes = gradient_bytes / len(participants)
        rounds = 2 * (len(participants) - 1)
        duration = 0.0
        used: set[str] = set()
        for _ in range(rounds):
            flows: list[tuple[LinkSpec, ...]] = []
            link_shares: Counter[str] = Counter()
            for index, source_id in enumerate(participants):
                target_id = participants[(index + 1) % len(participants)]
                source = self.nodes[source_id]
                target = self.nodes[target_id]
                path = self.route(source.region, target.region, link_states)
                if path is None:
                    return None
                flows.append(path)
                for link in path:
                    link_shares[link.link_id] += 1
                    used.add(link.link_id)

            round_seconds = 0.0
            for path in flows:
                if not path:
                    flow_seconds = 0.0
                else:
                    rate = min(
                        link.bytes_per_second / link_shares[link.link_id]
                        for link in path
                    )
                    propagation = sum(link.latency_seconds for link in path)
                    flow_seconds = propagation + chunk_bytes / rate
                round_seconds = max(round_seconds, flow_seconds)
            duration += round_seconds
        transferred = rounds * gradient_bytes
        return CollectiveEstimate(duration, frozenset(used), transferred)

    def transfer(
        self,
        source_region: str,
        target_region: str,
        payload_bytes: int,
        link_states: Mapping[str, LinkState],
    ) -> CollectiveEstimate | None:
        """Estimate one fixed-route transfer using its bottleneck link rate."""
        if payload_bytes < 0:
            raise ValueError("payload_bytes must be non-negative")
        path = self.route(source_region, target_region, link_states)
        if path is None:
            return None
        if not path or payload_bytes == 0:
            return CollectiveEstimate(0.0, frozenset(), payload_bytes)
        duration = sum(link.latency_seconds for link in path)
        duration += payload_bytes / min(link.bytes_per_second for link in path)
        return CollectiveEstimate(
            duration,
            frozenset(link.link_id for link in path),
            payload_bytes,
        )
