# WAN RL simulator status

## Current implementation

- The simulator advances a virtual clock through compute, WAN transfers,
  synchronous AllReduce, node membership changes, and link/node faults. It
  does not execute tensor or GPU operations.
- A policy can choose any currently feasible shard start, speculative duplicate,
  running-attempt cancellation, or wait from 0.1 to 10 seconds in 0.1-second
  increments. Starts and duplicates can use micro-batch sizes 32, 48, or 64.
- Run, batch, shard, node, and execution-attempt identities use UUIDs. Nodes
  can join, leave, and rejoin during an episode. A completed shard result stays
  buffered at the coordinator if its worker leaves during AllReduce.
- Episodes can now contain multiple sequential global batches. Virtual time,
  node/link state, and scheduled faults persist across batches; each batch gets
  fresh batch and shard UUIDs. The observation includes committed episode
  progress.
- The live RL worker mini-batch size is configurable using `RL_BATCH_SIZE` or
  `--batch-size`, with 32 as the default.

## Completed comparisons

The duplicate-aware one-batch comparison used 512 RLOO training episodes and
64 paired evaluation scenarios. RLOO was evaluated by seeded stochastic action
sampling. The episode-scale comparison used 64 scenarios per scale and ran
Fixed and Adaptive only; it did not train or evaluate RLOO.

| Batches per episode | Samples | Fixed success / mean time | Adaptive success / mean time |
| ---: | ---: | ---: | ---: |
| 1 | 768 | 98.4% / 2.67 s | 98.4% / 2.85 s |
| 4 | 3,072 | 100% / 9.15 s | 100% / 10.18 s |
| 16 | 12,288 | 100% / 33.11 s | 100% / 38.09 s |

Completion times are means of successful episodes. The 1-batch RLOO report and
checkpoint are from the earlier 1-batch observation format. The simulator's
new episode-progress observation changes the policy input size, so that
checkpoint must be retrained before it can run with multi-batch episodes. No
multi-batch RLOO training has been run. In the earlier 1-batch comparison,
RLOO completed 64/64 episodes in a mean 10.68 s, with 7.75 duplicates and 1.80
duplicate wins per episode. Fixed averaged 3.52 duplicates and 0.30 wins;
Adaptive averaged 0.64 duplicates and 0.42 wins.

Reports:

- `RL_schedule/results/wan-speculative-fixed-adaptive-rloo.json`
- `RL_schedule/results/wan-speculative-episode-scale-fixed-adaptive.json`

## Current model limits

- Node rates, topology, and fault schedules are synthetic examples, not
  measurements from a deployment.
- Input/result transfer contention is approximated by multiplying a new
  transfer's route time by one plus the busiest link's current in-flight flow
  count. Existing transfer finish times are not rescheduled.
- Routes are fixed. AllReduce is re-formed over surviving winner nodes after a
  membership fault; completed shard results are assumed buffered at the
  coordinator.
- The scale sweep adds node leave/rejoin waves as the episode grows. These are
  deterministic scenario events rather than sampled production traces.

## Follow-up work

1. Make the Adaptive baseline WAN-aware. Its current node-speed heuristic can
   interleave regions and increase ring AllReduce time; at scale 16 the mean
   collective time was 12.55 s for Adaptive and 6.53 s for Fixed.
2. Replace the input/result contention multiplier with per-link shared-capacity
   scheduling that updates in-flight completion times when flows join or leave.
3. Add measured node/network profiles and longer fault traces, then repeat the
   paired Fixed/Adaptive scale sweep across more seeds.
4. Define multi-batch failure and recovery semantics, including whether an
   interrupted global batch retries from its last committed boundary.
5. After approval to train, train a new RLOO v2 policy on multi-batch episodes.
   Compare success rate, time-to-target, goodput, duplicate efficiency, and
   collective/recovery overhead against both baselines.
6. Calibrate the synthetic compute and communication costs before making any
   real-system speedup claim.
