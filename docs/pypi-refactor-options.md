# PyPI refactor options

Research snapshot: 2026-10-24. This branch copies the current uncommitted
`pipeline-exec` worktree. No replacement dependency has been installed or
adopted yet. The project currently depends only on `torch>=2.2`.

The inspection environment has PyTorch 2.14.1+cpu and includes
`torch.distributed.pipelining`; `grpcio`, Ray, `safetensors`, and DiskCache are
not installed there.

## Findings

| What we own now | Candidate | What it could replace | Tradeoffs |
|---|---|---|---|
| TCP framing, client/worker RPC, separate heartbeat socket | `grpcio`, `grpcio-tools`, gRPC health service | `protocol.py` framing and much of the connection/heartbeat code; protobuf defines request schemas and streaming can carry tensor chunks | Keeps our stage runtime, network-aware scheduler, activation spool, and retry policy. It does not give durable job recovery or stage scheduling. Health state still has to be maintained by the service. |
| Coordinator, worker registry, task routing, resource scheduler, object transfer and spill | `ray` | Most of the custom coordinator/worker runtime; Ray tasks/actors, object store, cluster scheduling, retries and object spilling | Biggest rewrite and runtime addition. Ray nodes require bidirectional network access and multiple ports. Retry semantics can be at-least-once, and application checkpoints remain the application's responsibility. Object spilling restores an object to memory when consumed; it does not make one active operation fit in less memory. |
| FX graph cut, two-stage scheduling, microbatch pipeline, future backward scheduling | `torch.distributed.pipelining` (part of PyTorch) | Stage partitioning and pipeline schedules such as GPipe and 1F1B; it includes forward/backward pipeline machinery | Closest fit for distributed training, but currently alpha. It expects a `torch.distributed` process group and stage input/output metadata; automatic splitting uses `torch.export` tracing. It changes our coordinator/worker design rather than dropping into it. Check the supported minimum PyTorch version against our current `torch>=2.2`. |
| Tensor payload bundle / future disk checkpoints | `safetensors` | Named tensor file serialization, especially checkpoint and parameter files; Python loading can use memory mapping | Useful for safe tensor files, but it is not an RPC transport, queue, scheduler, or retry system. It does not by itself remove the 64 MiB wire-frame limit or the need for a worker to materialize the active tensor. |
| Small temporary activation spool | `diskcache` | Persistent disk-backed deque/cache if the queue needs cross-process persistence | More machinery than the current temporary-file spool. It may help with a durable job journal, but it does not solve active-tensor memory or chunked transfer. The current spool is already small and has bounded backpressure. |

## What the official docs establish

- gRPC Python supports unary, request-streaming, response-streaming, and
  bidirectional-streaming RPCs. The standard health service supports `Check`
  and streaming `Watch`, while the application updates each service's health
  state. Sources: [gRPC Python basics](https://grpc.io/docs/languages/python/basics/),
  [gRPC health checking](https://grpc.io/docs/guides/health-checking/),
  [grpcio on PyPI](https://pypi.org/project/grpcio/).
- Ray can restart actors and retry actor tasks when configured. Actor task
  retries may execute a method more than once; actor state recovery still
  requires application checkpoints. Ray spills object-store values to local
  or external storage under memory pressure and restores them for consumption.
  Its nodes require bidirectional communication and multiple ports. Sources:
  [Ray actor fault tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html),
  [Ray object spilling](https://docs.ray.io/en/latest/ray-core/objects/object-spilling.html),
  [Ray port configuration](https://docs.ray.io/en/latest/ray-core/configure.html),
  [Ray on PyPI](https://pypi.org/project/ray/).
- PyTorch `torch.distributed.pipelining` offers model splitting and established
  pipeline schedules, including cross-host use. It is currently documented as
  alpha; automatic graph splitting depends on `torch.export` traceability, and
  stage metadata is fixed after its first step. Sources:
  [PyTorch pipeline parallelism](https://docs.pytorch.org/docs/main/distributed/pipelining.html).
- `safetensors` is a named-tensor file format with a Python/PyTorch API and
  memory-mapped loading support. Sources:
  [safetensors on PyPI](https://pypi.org/project/safetensors/),
  [safetensors project](https://github.com/huggingface/safetensors).
- DiskCache provides a persistent disk-backed cache and deque. Sources:
  [DiskCache on PyPI](https://pypi.org/project/diskcache/),
  [DiskCache documentation](https://grantjenks.com/docs/diskcache/).

## Recommendation

There is no single small library that preserves all current behavior while
replacing every custom component.

1. For an incremental refactor that keeps the network experiments, prototype
   gRPC for the worker data and health-control APIs. Keep the network-aware
   scheduler, stage runtime, and bounded activation spool. Compare code size,
   throughput, and behavior under the existing latency/loss profiles.
2. For MNIST and distributed training, prototype
   `torch.distributed.pipelining` in a separate small example. Compare its
   microbatch outputs and gradients with the current stage runtime before
   replacing the custom pipeline.
3. Consider Ray only if the goal changes to a general distributed Python
   runtime and its bidirectional, multi-port cluster network fits deployment.
4. Do not replace the temporary spool with DiskCache yet. Consider it when
   durable cross-process job checkpoints are implemented. Consider
   safetensors for those persistent tensor files, not as a substitute for the
   transport.

## Proposed spikes

- **gRPC spike:** implement `Status`, `StageDefine`, `StagePut`, `StageRun`,
  `StageDrop`, and a long-lived health stream; preserve the existing stage
  implementation and compare behavior under 0%, 1%, and 10% packet loss.
- **PyTorch pipeline spike:** run the same two-stage model and microbatch count
  with `torch.distributed.pipelining` on two CPU worker processes; compare
  outputs, gradients, launch complexity, and whether the current topology's
  network controls remain usable.
- **Ray spike:** run the same model as Ray actors/tasks with object spilling
  configured, then measure startup cost, memory use, network reachability, and
  failure behavior before considering migration.
