"""Event-driven simulator for synchronous training over a fixed WAN topology."""

from .environment import SynchronousWANScheduler
from .examples import example_workload, four_region_topology
from .speculative import DecisionAction, DecisionKind, SpeculativeWANScheduler
from .types import (
    FaultEvent,
    FaultKind,
    LinkSpec,
    NodeSpec,
    NodeState,
    RecoveryConfig,
    SimulationConfig,
    StepResult,
    Topology,
    Workload,
)

__all__ = [
    "FaultEvent",
    "FaultKind",
    "DecisionAction",
    "DecisionKind",
    "LinkSpec",
    "NodeSpec",
    "NodeState",
    "RecoveryConfig",
    "SimulationConfig",
    "StepResult",
    "SynchronousWANScheduler",
    "SpeculativeWANScheduler",
    "Topology",
    "Workload",
    "example_workload",
    "four_region_topology",
]
