# FSSLK

FSSLK is a prototype for coordinating CPU work across machines. The command line uses gRPC for worker data and health streams, status, tensor sums, and pipeline stage calls. The previous length-prefixed TCP transport remains available with `--transport tcp`. gRPC currently uses insecure channels, so deploy it only on a trusted network.

## Run with Docker Compose

Build and start one coordinator plus fast, medium, and slow network-profile Workers:

```bash
docker compose up --build -d
```

Show the workers registered with the coordinator:

```bash
docker compose exec coordinator python -m fsslk status --host coordinator
```

Worker status reports `healthy`, `suspect`, or `unreachable` from its
acknowledged control-channel heartbeat. With the default 2-second heartbeat
interval and 15-second lease, a worker becomes suspect after 7.5 seconds without
a heartbeat and unreachable after 15 seconds. Stage data and health use
separate gRPC streams.

Split a CPU tensor across the workers and add the partial sums:

```bash
docker compose exec coordinator python -m fsslk run-sum --host coordinator --size 16
```

Stop the cluster:

```bash
docker compose down
```

## Run a two-worker model pipeline

The `run-pipeline` prototype captures the demo Transformer, splits its graph
into two stages, deploys one stage to each of two workers, and streams input
micro-batches through them. Worker IDs default to container hostnames, so read
the IDs and check that both workers are active before starting:

```bash
docker compose up -d --build
FAST=$(docker compose exec -T worker-fast hostname)
MED=$(docker compose exec -T worker-medium hostname)
docker compose exec -T coordinator python -m fsslk status
```

Run the default model with one micro-batch or four. The coordinator writes each
measurement file inside its container:

```bash
docker compose exec -T coordinator python -m fsslk run-pipeline \
  --workers "$FAST,$MED" --repeats 3 --json /tmp/run_m1.json
docker compose exec -T coordinator python -m fsslk run-pipeline \
  --workers "$FAST,$MED" --microbatches 4 --repeats 3 --json /tmp/run_m4.json
```

For a single forward pass without timing output, use `run-inference`. It returns
the final model output values; intermediate activations stay inside the
coordinator pipeline. Pass `--json` to write those values to a file:

```bash
docker compose exec -T coordinator python -m fsslk run-inference \
  --workers "$FAST,$MED" --layers 1 --dim 16 --heads 4 --batch 1 --seq 8 \
  --json /tmp/output.json
```

Copy the JSON results to the host if needed:

```bash
docker compose cp coordinator:/tmp/run_m1.json /tmp/run_m1.json
docker compose cp coordinator:/tmp/run_m4.json /tmp/run_m4.json
```

`--microbatches` splits the input on its batch axis. Stage A and stage B process
different chunks at the same time; it does not split a single matrix operation
across workers. Each matrix and its weights stay whole on the worker assigned
to that stage. Stage activations pass through the coordinator.

This is a CPU prototype with two stages and two workers. It accepts one tensor
input and only operations from the worker allowlist; the wire format currently
supports CPU `int64` and `float32` tensors up to 64 MiB per tensor frame by
default. Set `FSSLK_MAX_PAYLOAD_BYTES` on all participants to change that frame
size. Models must be in evaluation mode. Pipeline stage calls do not have
application-level retry, and network transfer can cost much more than
computation on slow or lossy links. The reported link measurements are
captured before deployment and are not refreshed during each measured run.

When stage B is slower than stage A, the coordinator writes queued A-to-B
activation payloads to temporary files and keeps at most one activation waiting
in the handoff queue; stage A pauses while that queue is full. Temporary files
are removed as stage B reads them and when a run exits. Set
`FSSLK_ACTIVATION_SPOOL_DIR` to choose a mounted disk-backed directory instead
of the system temporary directory. The active micro-batch and final model output
still need memory, and one frame is still limited to the configured payload
size (64 MiB by default); larger activations need chunked transfer and
tile-aware operations.

## Stage-level automatic differentiation

`StageRuntime.run_with_grad(inputs)` runs one stage with PyTorch autograd enabled
and returns a `StageExecution`. Call `execution.backward(output_gradients)` to
get gradients for that execution's floating-point inputs and weights. The
caller can pass an input gradient from one stage to the previous stage, which
is the basic vector-Jacobian-product needed for pipeline backpropagation.
The current stage spec does not distinguish trainable parameters from floating
buffers, so all floating-point stage params are treated as gradient targets.

This is only a local stage primitive. The distributed pipeline still runs in
evaluation mode; there is no network backward command, loss calculation,
optimizer update, gradient aggregation, or MNIST training loop yet.

Workers retry their gRPC data stream while the coordinator starts and reconnect
their acknowledged health stream independently. For a local non-Docker run,
start `python -m fsslk coordinator`, then start one or more
`python -m fsslk worker --host 127.0.0.1` processes in separate terminals. Use
`python -m fsslk status` and `python -m fsslk run-sum --size 16` from another
terminal. Add `--transport tcp` to all participants to use the previous
transport instead.

## Remote PyTorch callables

`fsslk.fsslk_ddp` wraps a function or callable class. Without `axis`, each
invocation runs on one healthy Worker; calls are assigned round-robin unless
you select a `worker`. With `axis=0`, the first positional tensor is split
into up to four chunks per healthy Worker. Chunk dispatch order is shuffled;
each Worker gets another chunk as soon as it finishes its current one. Results
are reassembled in the original axis order and concatenated. Choose this only
when the operation is independent across that axis. Use `merge="sum"` for
operations that return partial sums.
Configure the same token in the Coordinator and client environments before
using this API:

```bash
export FSSLK_DDP_TOKEN='use-a-long-random-value'
python -m fsslk coordinator
```

Start one or more Workers as usual. Then call a wrapped function from a client
process that has `FSSLK_DDP_TOKEN` set:

```python
import torch
import fsslk

@fsslk.fsslk_ddp(axis=0)
def relu(values):
    return torch.relu(values)

result = relu(torch.tensor([-1.0, 2.0]))
```

Callable classes can be decorated as well:

```python
@fsslk.fsslk_ddp(axis=0)
class Scale(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, values):
        return values * self.factor

scale = Scale(3)
result = scale(torch.tensor([1.0, 2.0]))
```

A class instance is created on the Worker for each call, so constructor and
call arguments must be serializable. Remote callables and results use
cloudpickle. Their serialized bytes are sent in chunks of at most 64 MiB by
default; the total limit defaults to 1 GiB. Set `FSSLK_MAX_PAYLOAD_BYTES` to
change the chunk size and `FSSLK_MAX_REMOTE_CALL_BYTES` to change the total
limit. Use the same settings on the client, Coordinator, and Workers. The
receiver reassembles the full payload in memory before deserializing it.
This API runs supplied Python code on Workers; use it only with trusted clients
and a trusted network. gRPC traffic is not encrypted. Returned tensors are
values and do not carry an autograd graph back to the client.

## Run regression tests

```bash
docker compose run --build --rm --no-deps \
  -v "$PWD/tests:/tests:ro" \
  --entrypoint python coordinator \
  -m unittest discover -s /tests -v
```

## Prototype limits

- This version uses CPU-only PyTorch; it does not require a GPU.
- Tensor RPCs carry protobuf metadata and raw tensor bytes. The current codec
  supports CPU `int64` and `float32` tensors, with a 64 MiB maximum payload per
  frame by default (`FSSLK_MAX_PAYLOAD_BYTES`). Remote Python calls stream
  larger serialized payloads in frames and use a configurable total limit
  (`FSSLK_MAX_REMOTE_CALL_BYTES`, 1 GiB by default).
- The legacy TCP transport uses a length-prefixed JSON header followed by raw tensor bytes. Coordinator, Worker, and CLI must use the same transport; gRPC and TCP cannot share one coordinator port.
- `status` reports CPU sum throughput, rolling RTT and jitter, coordinator-to-worker transfer rate, UDP probe packet loss, and recent worker probe success. RTT probes run every 5 seconds; a 256 KiB bandwidth probe and 100-packet UDP loss probe run every 30 seconds.
- Packet loss is measured on the UDP probe path and can differ from TCP data loss. The Compose setup publishes UDP port 27495; other deployments must allow the coordinator's UDP probe port.
- Compose applies different Linux `netem` profiles to the Workers: fast uses 2 ms / 500 Mbps / 0% loss, medium 25 ms / 50 Mbps / 1%, and slow 75 ms / 10 Mbps / 3%. Each profile is applied in both traffic directions.
- Work chunks use a deterministic completion-time heuristic combining CPU sum throughput, measured transfer rate, UDP probe loss, and RTT. It does not use reinforcement learning; missing network measurements fall back to compute-only weighting.
- If a worker disconnects, fails a sum task, or does not respond within 30 seconds, the coordinator retries that chunk on another active worker. A chunk gets at most three attempts, and a late response from an earlier attempt is ignored.
- Retries currently apply only to the prototype's stateless `sum` operation. If no other active worker is available, the request fails.
- `FSSLK_DDP_TOKEN` guards remote callable execution. Other coordinator APIs are unauthenticated, and gRPC traffic is plaintext.
