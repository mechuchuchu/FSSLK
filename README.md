# FSSLK

FSSLK is a prototype for coordinating CPU work across machines. This first version connects a coordinator to CPU workers over TCP, tracks worker heartbeats, and distributes a small PyTorch tensor sum.

## Run with Docker Compose

Build and start one coordinator and two workers:

```bash
docker compose up --build --scale worker=2 -d
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

## Prototype limits

- This version uses CPU-only PyTorch; it does not require a GPU.
- Tensor values travel as JSON numbers, which is suitable only for small demonstrations.
- Workers are selected in connection order. Network-aware scheduling, authentication, encryption, and WAN tuning are not implemented yet.
