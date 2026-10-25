# RL scheduling experiments

This directory contains the implementation and design notes for the staged RL
scheduling study. The experiments can run independently of the FSSLK runtime.

## Step 1: single-process MLP baseline

The first step trains a small MLP to fit fixed random input and target vectors.
The dataset and initial model use separate, recorded seeds. Each run records
epoch losses, optimizer-step times, time to the target loss, and SHA-256
fingerprints for the dataset, initial full-dataset gradient, and final model.

Run it from the repository root:

```bash
python RL_schedule/mlp_baseline.py --json results/mlp-baseline.json
```

Defaults follow the document's example dataset dimensions (1,024 samples,
32 input values, and 16 target values). The hidden width, epoch limit, batch
size, learning rate, target loss, and seed can be changed with command-line
options. If the target loss is not reached within the epoch limit, the run
still writes its final loss and timing record.

## Step 2: exact shard-gradient aggregation

Split the fixed dataset into shards, calculate each shard's gradient from the
same seeded model, and compare the weighted aggregate with the single-process
gradient:

```bash
python -m RL_schedule.distributed_gradient --num-shards 4
```

Every shard is evaluated against the same seeded model parameters. Since each
shard's MSE is a mean, its gradient is weighted by `shard_size / sample_count`
before aggregation. The report includes shard sizes, maximum absolute error,
L2 error, and fingerprints for both gradients. These values provide a numeric
comparison; floating-point summation order can make the fingerprints differ
even when the gradients agree within a small numeric tolerance.

## Step 3: worker and network simulator

The simulator models compute time, input and result transfer time, worker
progress, heartbeats, stalls, failures, and lost results. It models one delayed
shard so the WAIT and DUPLICATE control loop can be studied independently; the
live coordinator applies those decisions across multiple training shards. The
reference policies can be compared with:

```bash
python -m RL_schedule.evaluate --episodes 100 --seed 0
```

The report includes completion rate, makespan, network work, and duplicate
compute. The adaptive heuristic and fixed-timeout policy use the same seeded
initial worker for each episode. The simulator is pure Python/PyTorch and does
not require Docker; Docker is used in Step 5 for real-worker validation.

The static Equal Split and Proportional assignment baselines are available
separately:

```bash
python -m RL_schedule.allocators --shards 32
```

## Step 4: train and compare RL policies

PPO trains against randomized worker conditions. History length 1 is the
instantaneous policy; history length 4 gives the policy the recent sequence of
worker observations. Evaluation uses fixed seen and held-out worker conditions
and reports the heuristic baselines alongside the learned policy.

```bash
python -m RL_schedule.rl_policy train \
  --history-length 1 --output RL_schedule/results/ppo-instant.pt --eval-episodes 50
python -m RL_schedule.rl_policy train \
  --history-length 4 --output RL_schedule/results/ppo-history.pt --eval-episodes 50
python -m RL_schedule.rl_policy train \
  --episodes 5000 --architecture lstm --history-length 4 \
  --output RL_schedule/results/ppo-lstm.pt --eval-episodes 100
python -m RL_schedule.rl_policy train-rloo \
  --episodes 5000 --group-size 8 \
  --output RL_schedule/results/rloo-lstm.pt --eval-episodes 100
python -m RL_schedule.rl_policy train \
  --worker-count 6 --history-length 4 \
  --output RL_schedule/results/ppo-history-6w.pt --eval-episodes 50
python -m RL_schedule.rl_policy train \
  --episodes 4000 --resume-checkpoint RL_schedule/results/ppo-history.pt \
  --output RL_schedule/results/ppo-history-5k.pt --eval-episodes 100
python -m RL_schedule.rl_policy compare \
  --instantaneous RL_schedule/results/ppo-instant.pt \
  --history-aware RL_schedule/results/ppo-history.pt --episodes 100
```

The policy observes only simulator telemetry. It chooses between short,
medium, or long waits and duplicating the current shard on an available worker.
The default `cost_sensitive` reward charges elapsed time, network work,
duplicate compute, and failed episodes, while evaluation reports makespan and
resource costs separately.
`--worker-count` selects the fleet size for training and evaluation; each
checkpoint is tied to that count because its observation and action dimensions
depend on the worker list. The current simulator scales the scheduling choices
for one delayed shard, so this option compares policy behavior across fleet
sizes and does not measure multi-shard training throughput.
`--architecture lstm` feeds the recent observation frames in order through an
LSTM actor-critic. Its recurrent state is recomputed from the fixed observation
window at each decision; it is not carried between decisions.
`train-rloo` uses an actor-only LSTM whose hidden state persists across
decisions within an episode. It samples groups of trajectories under the same
worker condition and uses each trajectory's return minus the other group
members' mean return as its RLOO advantage. It performs one on-policy update
per group and does not train a value critic.
To train RLOO with only elapsed time and binary success in the reward, pass
`--reward-mode time_success`. Each step receives `-elapsed_seconds`, with `+1`
added on successful episode completion; there is no separate network, duplicate
compute, or failure penalty in this mode.
For counts above three, extra simulator profiles are seeded variations of the
fast, medium, and slow templates. These vary the simulator's training
conditions; the live Compose fleet separately defines and measures workers
4–6 using the fixed `seen` profiles.
`--resume-checkpoint` continues from a compatible policy. Checkpoints that do
not contain optimizer state resume with a freshly initialized Adam optimizer.

## Step 5: schedule real worker tasks

The Docker Compose setup defines six isolated CPU workers with different
network profiles; the default coordinator run still starts the original three.
For scaling runs, the coordinator uses a fixed `seen` profile list matching
the endpoint count. It sends each worker its fixed data shard and
the same model version. It polls task progress and heartbeat age, then uses
the selected live scheduler to wait or duplicate a pending shard on an
available worker. The first accepted result for each shard version is weighted
by shard size; late copies are ignored and asked to stop before the next model
update. The first distributed gradient is compared with the single-process
reference.

```bash
docker compose -f RL_schedule/compose.yaml up --build \
  --abort-on-container-exit --exit-code-from coordinator
```

The run record is written to `RL_schedule/results/live-run.json`. Worker
services `worker-4` through `worker-6` are available for scaling runs. RLOO
checkpoints are tied to their worker count, so train a matching checkpoint for
each fleet size before a live policy run:

```bash
python -m RL_schedule.rl_policy train-rloo --worker-count 4 \
  --output RL_schedule/results/rloo-lstm-4w.pt
python -m RL_schedule.rl_policy train-rloo --worker-count 6 \
  --output RL_schedule/results/rloo-lstm-6w.pt
```

`RL_schedule.live_compare` runs paired repeated live trials for an existing
cost-sensitive RLOO checkpoint, a time-success RLOO checkpoint, adaptive, and
fixed policies. It shuffles policy order per repetition and writes running
means, confidence intervals, paired differences, and each individual trial.
Run it inside the Compose network with `docker compose run --entrypoint python`
after starting the requested worker services.

Set `RL_STEPS=20` to shorten a run. The default `adaptive` scheduler uses the
heuristic policy; `RL_SCHEDULER=fixed` selects the fixed-timeout baseline.
Train a history-aware PPO checkpoint locally, then use it with the same
Compose workers:

```bash
python -m RL_schedule.rl_policy train \
  --history-length 4 \
  --output RL_schedule/results/ppo-history.pt \
  --eval-episodes 50
RL_SCHEDULER=ppo \
RL_POLICY_CHECKPOINT=/results/ppo-history.pt \
docker compose -f RL_schedule/compose.yaml up --build \
  --abort-on-container-exit --exit-code-from coordinator
```

The worker protocol is JSON metadata plus base64 float32 tensors; it does not
deserialize Python objects or execute uploaded code. Each response includes a
model version, shard ID, attempt ID, and task ID. The coordinator validates
those fields and accepts at most one finite, correctly shaped gradient for
each shard before updating the model. The synchronous `distributed_train`
runner remains available for checking basic shard aggregation without live
scheduling.
