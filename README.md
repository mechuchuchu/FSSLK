# FSSLK

FSSLK is a prototype for coordinating CPU work across machines. This first version connects a coordinator to CPU workers over TCP, tracks worker heartbeats on a separate control connection, and distributes a small PyTorch tensor sum.

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
a heartbeat and unreachable after 15 seconds. Stage data continues over the
separate data connection.

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
supports CPU `int64` and `float32` tensors up to 64 MiB per tensor frame. Models
must be in evaluation mode. Pipeline stage calls do not have application-level
retry, and network transfer can cost much more than computation on slow or
lossy links. The reported link measurements are captured before deployment and
are not refreshed during each measured run.

When stage B is slower than stage A, the coordinator writes queued A-to-B
activation payloads to temporary files and keeps at most one activation waiting
in the handoff queue; stage A pauses while that queue is full. Temporary files
are removed as stage B reads them and when a run exits. Set
`FSSLK_ACTIVATION_SPOOL_DIR` to choose a mounted disk-backed directory instead
of the system temporary directory. The active micro-batch and final model output
still need memory, and one frame is still limited to 64 MiB; larger activations
need chunked transfer and tile-aware operations.

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

The worker retries its connection while the coordinator starts. For a local non-Docker run, start `python -m fsslk coordinator`, then start one or more `python -m fsslk worker --host 127.0.0.1` processes in separate terminals. Use `python -m fsslk status` and `python -m fsslk run-sum --size 16` from another terminal.

## Run regression tests

```bash
docker compose run --build --rm --no-deps \
  -v "$PWD/tests:/tests:ro" \
  --entrypoint python coordinator \
  -m unittest discover -s /tests -v
```

## Prototype limits

- This version uses CPU-only PyTorch; it does not require a GPU.
- Tensor frames use a length-prefixed JSON header followed by raw tensor bytes. The current codec supports CPU `int64` and `float32` tensors, with a 64 MiB maximum payload per frame.
- Coordinator, Worker, and CLI must use the same version; this frame format is not compatible with the previous newline-delimited JSON protocol.
- `status` reports CPU sum throughput, rolling RTT and jitter, coordinator-to-worker transfer rate, UDP probe packet loss, and recent TCP probe success. RTT probes run every 5 seconds; a 256 KiB bandwidth probe and 100-packet UDP loss probe run every 30 seconds.
- Packet loss is measured on the UDP probe path and can differ from TCP data loss. The Compose setup publishes UDP port 27495; other deployments must allow the coordinator's UDP probe port.
- Compose applies different Linux `netem` profiles to the Workers: fast uses 2 ms / 500 Mbps / 0% loss, medium 25 ms / 50 Mbps / 1%, and slow 75 ms / 10 Mbps / 3%. Each profile is applied in both traffic directions.
- Work chunks use a deterministic completion-time heuristic combining CPU sum throughput, measured transfer rate, UDP probe loss, and RTT. It does not use reinforcement learning; missing network measurements fall back to compute-only weighting.
- If a worker disconnects, fails a sum task, or does not respond within 30 seconds, the coordinator retries that chunk on another active worker. A chunk gets at most three attempts, and a late response from an earlier attempt is ignored.
- Retries currently apply only to the prototype's stateless `sum` operation. If no other active worker is available, the request fails.
- Authentication and encryption are not implemented yet.
