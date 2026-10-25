# Synchronous WAN simulator

This package models synchronous data-parallel training without running tensor
operations or GPU kernels. It is separate from the single delayed-shard
scheduling simulator in `RL_schedule/simulator.py` and has no dependency on a
particular RL algorithm or Gymnasium.

The caller supplies a local sample allocation for each global batch. The
simulator models fixed node FLOP rates, local compute completion, a synchronous
barrier, ring AllReduce over a fixed region topology, and commit only after the
collective succeeds. A failed or interrupted step does not increase the
committed sample count.

The event engine uses virtual time. Same-time fault events have priority over
compute and collective completion events. Each compute task and collective has
a generation number so an invalidated completion cannot commit later. Node
speeds remain fixed unless an explicit event puts a node into `near_zero`; that
state uses the configured small speed fraction. Nodes can also stall, fail,
leave, recover, and rejoin. WAN links have fixed bandwidth and latency while
up, and can be taken down and brought back by scheduled events.

```python
from RL_schedule.wan_sim import FaultEvent, FaultKind, SimulationConfig
from RL_schedule.wan_sim.environment import SynchronousWANScheduler
from RL_schedule.wan_sim.examples import example_workload, four_region_topology

simulator = SynchronousWANScheduler(
    topology=four_region_topology(),
    workload=example_workload(),
    config=SimulationConfig(),
    fault_schedule=(
        FaultEvent(0.5, FaultKind.NODE_SPEED_DROP, "node-8", duration_seconds=3.0),
        FaultEvent(8.0, FaultKind.NODE_LEAVE, "node-3", duration_seconds=12.0),
    ),
)

result = simulator.step(
    {f"node-{index}": 96 for index in range(1, 9)},
    {f"node-{index}": (32, 48, 64)[(index - 1) % 3] for index in range(1, 9)},
)
print(result)
print(simulator.state())
print(simulator.metrics())
```

`four_region_topology()` contains illustrative values only. Replace its node
rates, memory limits, regional paths, bandwidths, and latencies with measured
values before using the simulator for performance claims. Fault schedules are
deterministic and expressed in absolute episode seconds, which makes scenarios
reproducible. The current core returns a failed step to the caller; a policy or
scenario runner decides how to reallocate and retry the uncommitted global
batch.

## Current model boundaries

- Compute time is `local_samples * flops_per_sample / node_flops_per_second`.
- Optional per-node micro-batch sizes add a configured per-micro-batch launch
  overhead and an estimated activation-memory footprint. The simulator accepts
  the selected sizes but does not choose them; a future policy adapter can
  provide them as part of an action.
- Ring AllReduce runs `2 * (participants - 1)` rounds. Concurrent flows share
  each traversed link equally within a round. Routes are chosen once from the
  base topology by propagation latency; a down link on a fixed route interrupts
  the collective instead of triggering rerouting.
- Faults are explicit events. This version does not sample fault distributions,
  simulate checkpoint files, model optimizer state, or run actual collective
  protocols.
- `compute_seconds` is summed node busy time (resource-seconds); barrier wait is
  the sum of each participant's wait for the slowest local compute.
- The API accepts a feasible sample allocation. It does not choose allocations,
  communication priorities, retry policies, or RL actions.

An optional Gymnasium adapter provides a replaceable starter action encoding.
It accepts continuous per-node allocation weights and discrete per-node
micro-batch choices, then projects the weights onto a feasible integer global
batch. The adapter is not part of the simulator core:

```python
from RL_schedule.wan_sim.gym_env import GymnasiumSynchronousWANEnv

env = GymnasiumSynchronousWANEnv(four_region_topology(), example_workload())
observation, info = env.reset(seed=0)
```

Install the research adapter dependency with `uv pip install gymnasium`.

## Preliminary batch-allocation comparison

Run a paired policy comparison with:

```bash
python -m RL_schedule.wan_sim.compare \
  --episodes 2048 --group-size 8 --eval-episodes 128 --seed 0
```

This first comparison uses the simpler `GymnasiumSynchronousWANEnv`: fixed
equal splits, speed-weighted splits, and a small RLOO allocator. It does not
include speculative duplicates and is retained as a batch-allocation baseline.

## Duplicate-aware fixed, adaptive, and RLOO comparison

Use the duplicate-aware candidate-action environment for the main comparison:

```bash
python -m RL_schedule.wan_sim.speculative_compare \
  --episodes 512 --group-size 8 --eval-episodes 64 --seed 0
```

Each global batch is divided into logical shards with stable UUIDs. Every
execution attempt receives a distinct UUID. The first valid result wins for a
shard; other in-flight attempts are cancelled and counted as ignored results.
Nodes can start outside membership, join during the episode, leave, and
rejoin. The default comparison starts six of eight potential nodes and
introduces the other two through scheduled join events.

At every decision point, RLOO scores all feasible candidates: start any pending
shard on any idle active node with a selected micro-batch, duplicate any
running shard on an idle node (including a different micro-batch), cancel any
running attempt, or wait from 0.1 to 10 seconds in 0.1-second increments.
Candidate generation enforces memory, membership, and one-attempt-per-node
constraints. The environment charges a configurable 1 ms scheduler decision
overhead, so dispatch/cancel loops cannot consume zero virtual time. Fixed and
adaptive baselines use the same candidates and event traces. RLOO evaluation
samples from its learned action distribution with a per-episode seed; it does
not switch to greedy argmax at evaluation time.

The fixed baseline starts shards in node order at micro-batch 32 and duplicates
a single-copy shard after one second. Adaptive assigns work to the fastest
effective node at micro-batch 64 and duplicates when its estimated completion
time is at least 0.1 seconds shorter.

The report includes completion time and rate, duplicate wins, ignored results,
WAN transfer and collective time, barrier wait, and per-policy action counts.
Each episode completes one global batch composed of logical shards. The 256
decision limit is a configurable episode guard (`--max-decisions`); reaching it
counts as an unsuccessful run. It writes JSON and a policy checkpoint under
`RL_schedule/results/`.

The included topology and cost values are synthetic examples. Micro-batch
timing uses configured launch overhead. Shard input and result transfers use
fixed shortest-latency routes and bottleneck bandwidth; concurrent AllReduce
flows share WAN link capacity. Calibrate these values and the transfer-sharing
model before interpreting a policy win as a real system speedup. A completed
shard result is treated as buffered at the coordinator; if a winner node leaves
during AllReduce, the collective is re-formed over the surviving winner nodes.
Concurrent shard input/result transfers use a lightweight contention estimate:
each new flow's route time is multiplied by one plus the busiest link's current
in-flight flow count. Existing transfer finish times are not rescheduled, so
this remains an approximation rather than a packet or fluid-level WAN model.

## Episode scale sweep without RLOO training

An episode can contain several sequential global batches. Virtual time,
membership, link state, and fault events continue across batch boundaries; each
batch and its shards receive new UUIDs. Run a fixed/adaptive-only scale sweep
with:

```bash
python -m RL_schedule.wan_sim.speculative_compare \
  --episode-scales 1 4 16 --eval-episodes 64 --seed 0
```

Passing `--episode-scales` skips RLOO training and evaluation. The decision
limit is multiplied by the scale so larger episodes retain the same per-batch
guard. Virtual time avoids waiting for simulated seconds, though simulator CPU
work still grows with the number of events and decisions. The observation now
includes episode progress; the saved one-batch RLOO checkpoint predates this
feature and needs retraining before it can run against multi-batch episodes.
