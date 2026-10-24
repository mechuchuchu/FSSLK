# FSSLK

FSSLK is a prototype for coordinating CPU work across machines. This first version connects a coordinator to CPU workers over TCP, tracks worker heartbeats, and distributes a small PyTorch tensor sum.

## Run with Docker Compose

Build and start one coordinator plus fast, medium, and slow network-profile Workers:

```bash
docker compose up --build -d
```

Show the workers registered with the coordinator:

```bash
docker compose exec coordinator python -m fsslk status --host coordinator
```

Split a CPU tensor across the workers and add the partial sums:

```bash
docker compose exec coordinator python -m fsslk run-sum --host coordinator --size 16
```

Stop the cluster:

```bash
docker compose down
```

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
