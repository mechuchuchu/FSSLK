# FSSLK — Final Goal & Development Plan

## 1. Project Overview

**FSSLK** is a PyTorch-oriented distributed computing runtime designed to utilize heterogeneous computing resources connected through the Internet.

Unlike traditional distributed training systems that are primarily optimized for stable, high-bandwidth LAN or datacenter environments, FSSLK is designed around a **WAN-first model** where participating machines may have different:

- CPU/GPU capabilities
- Network bandwidth
- Latency
- Jitter
- Packet loss
- Available memory
- Compute load
- Connection stability

The core idea is simple:

> **FSSLK continuously observes the computing and networking environment and automatically chooses an efficient execution, communication, and synchronization strategy.**

The goal is to make distributed computing across Internet-connected machines feel as close as possible to using a single computing environment.

---

# 2. Final Goal

The final user experience should look approximately like this.

### Coordinator

```bash
export FSSLK_PORT=27494
export FSSLK_MIN_INTERNET_SPEED=8000000000

fsslk --start
```

### Worker

```bash
fsslk --join <coordinator>:27494
```

After a worker joins, FSSLK should automatically perform:

```text
Node Discovery
      ↓
Authentication
      ↓
Network Benchmark
      ↓
Hardware Capability Detection
      ↓
Node Registration
      ↓
Scheduling
      ↓
Communication Optimization
      ↓
PyTorch Integration
      ↓
Distributed Execution
```

The user should not need to manually decide:

```text
Which node should execute this workload?
How much data should be transferred?
When should synchronization happen?
Should tensors be compressed?
Which nodes should participate?
What should happen if a node becomes slow?
```

FSSLK should make these decisions automatically whenever possible.

---

# 3. Core Design Philosophy

## 3.1 FSSLK is not simply an RPC replacement

FSSLK should not attempt to reinvent RPC.

RPC is useful for:

- Remote function execution
- Asynchronous execution
- Remote objects
- Distributed workflows

FSSLK focuses on a different layer:

```text
RPC
 └── Remote execution

FSSLK
 ├── Network awareness
 ├── Hardware awareness
 ├── Dynamic scheduling
 ├── WAN optimization
 ├── Tensor communication optimization
 ├── Node management
 └── Failure handling
```

FSSLK may use RPC where appropriate, while providing a specialized runtime for WAN-aware distributed computation.

---

# 4. High-Level Architecture

```text
                       PyTorch Application
                              │
                              ▼
                   ┌─────────────────────┐
                   │   FSSLK PyTorch API │
                   └──────────┬──────────┘
                              │
              ┌───────────────┴────────────────┐
              │                                │
              ▼                                ▼
      Custom Device Backend          Distributed Backend
         / Device Layer                 / ProcessGroup
              │                                │
              └───────────────┬────────────────┘
                              ▼
                   ┌─────────────────────┐
                   │    FSSLK Runtime    │
                   ├─────────────────────┤
                   │ Scheduler           │
                   │ Network Monitor     │
                   │ Hardware Monitor    │
                   │ Tensor Planner      │
                   │ Compression         │
                   │ Failure Manager     │
                   └──────────┬──────────┘
                              │
                   FSSLK Communication Layer
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
       Node A              Node B              Node C
       CPU/GPU             CPU/GPU             CPU/GPU
```

The architecture should keep the following concepts separate:

```text
Device Backend
    → computation and device abstraction

Distributed Backend
    → inter-node communication

FSSLK Runtime
    → scheduling, monitoring, optimization and coordination
```

---

# 5. Development Roadmap

## Phase 0 — Minimal Prototype

### Goal

Prove that multiple Internet-connected machines can join a FSSLK cluster and execute basic remote workloads.

### Components

- Coordinator
- Worker
- Node ID
- Authentication token
- TCP/QUIC transport
- Heartbeat
- Basic RPC
- Basic tensor transfer

Example:

```bash
fsslk --start
```

Worker:

```bash
fsslk --join 192.168.1.10:27494
```

Expected output:

```text
[FSSLK] Coordinator started
[FSSLK] Node joined: node-01
[FSSLK] Node joined: node-02
[FSSLK] Cluster size: 3
```

---

# 6. Phase 1 — Network Awareness

FSSLK should continuously monitor the network environment.

Measurements include:

```text
Bandwidth
RTT
Jitter
Packet Loss
Connection Stability
```

Example:

```text
Node 01
Bandwidth : 9.8 Gbps
RTT       : 4 ms
Loss      : 0.02%

Node 02
Bandwidth : 2.1 Gbps
RTT       : 71 ms
Loss      : 0.8%

Node 03
Bandwidth : 8.7 Gbps
RTT       : 9 ms
Loss      : 0.04%
```

Instead of simply checking:

```text
Bandwidth > 8 Gbps
```

FSSLK should eventually calculate a broader network score:

```text
Network Score =
    f(bandwidth,
      RTT,
      jitter,
      packet loss,
      stability)
```

This score becomes an input to the scheduler.

---

# 7. Phase 2 — Hardware Awareness

FSSLK should detect the computing capabilities of each worker.

Possible measurements:

```text
CPU cores
CPU throughput
GPU type
GPU memory
GPU throughput
RAM
Available RAM
Available VRAM
Current system load
```

Example:

```text
Node A
GPU: GPU-A
Compute: 100 TFLOPS
VRAM: 24 GB

Node B
GPU: GPU-B
Compute: 35 TFLOPS
VRAM: 12 GB
```

FSSLK should use both measured hardware capability and current workload rather than relying only on static hardware specifications.

---

# 8. Phase 3 — Adaptive Scheduler

This is one of the core components of FSSLK.

The scheduler should consider:

```text
Compute Capability
+
Network Capability
+
Current Load
+
Memory Availability
+
Communication Cost
+
Task Characteristics
```

For example:

```text
Total workload = 100

Node A → 50
Node B → 10
Node C → 40
```

The allocation should not necessarily remain fixed.

If Node C's network becomes unstable, FSSLK could dynamically reduce the amount of communication-heavy work assigned to that node.

---

# 9. Phase 4 — Tensor Communication Engine

WAN communication can become the primary bottleneck in distributed training.

FSSLK should therefore provide an optimized tensor communication layer.

Potential features:

```text
Tensor chunking
Parallel transfer
Compression
Quantization
Caching
Prefetching
Pipelining
Transfer scheduling
```

Conceptually:

```text
Large Tensor
     │
     ├── Chunk 1 ─┐
     ├── Chunk 2 ─┤
     ├── Chunk 3 ─┼── Network
     └── Chunk 4 ─┘
```

For suitable workloads, FSSLK could optionally use:

```text
FP32
 ↓
FP16 / BF16 / INT8
 ↓
Transfer
 ↓
Reconstruction
```

The system must measure the trade-off between communication savings, computation overhead, and model accuracy.

---

# 10. Phase 5 — PyTorch Distributed Backend

FSSLK should integrate with PyTorch's distributed architecture.

A primary goal is to implement an FSSLK ProcessGroup capable of supporting operations such as:

```text
all_reduce
all_gather
reduce_scatter
broadcast
send
recv
```

The first implementation should prioritize correctness and reliability.

Later versions can introduce:

```text
Adaptive All-Reduce
Compressed All-Reduce
Hierarchical Communication
WAN-Aware Synchronization
Communication Scheduling
```

The key idea is to make collective communication aware of WAN conditions instead of treating every connection as equivalent.

---

# 11. Phase 6 — PyTorch Custom Device Integration

Where appropriate, FSSLK can use PyTorch's custom device/backend extension mechanisms.

The long-term goal is to provide a natural integration between FSSLK and PyTorch.

Conceptually:

```text
PyTorch
   │
   ├── Device Backend
   │
   └── Distributed Backend
             │
             ▼
          FSSLK
```

The implementation should avoid unnecessarily coupling device abstraction and distributed communication.

A custom device represents a computation environment.

A distributed backend represents communication between environments.

FSSLK coordinates both.

---

# 12. Phase 7 — Automatic Optimization

The ultimate objective is for FSSLK to automatically determine:

```text
Where computation should happen
What should be transferred
When data should be transferred
How data should be encoded
When synchronization should occur
Which nodes should participate
```

Example:

```text
Node A
Fast GPU
Fast network
→ Large workload

Node B
Slow GPU
Fast network
→ Small compute workload / communication-heavy role

Node C
Fast GPU
Slow network
→ More local computation
```

The scheduler continuously adapts to changing conditions.

---

# 13. Phase 8 — Fault Tolerance

Internet-based distributed systems must assume that nodes can disappear.

FSSLK should handle:

```text
Node disconnect
Node timeout
Bandwidth degradation
Packet loss increase
Process crash
GPU failure
Network partition
Temporary connectivity loss
```

Example:

```text
Node B
   ↓
Connection lost
   ↓
FSSLK detects failure
   ↓
Scheduler recalculates cluster state
   ↓
Remaining nodes receive redistributed work
```

The long-term goal is to prevent a single unreliable worker from unnecessarily terminating the entire distributed workload.

---

# 14. Phase 9 — Security

Because FSSLK is designed for Internet-connected nodes, security is a fundamental requirement.

Required features include:

```text
Authentication
Authorization
Encrypted transport
Node identity
Token management
Replay protection
Rate limiting
Resource limits
```

A simplified workflow:

```text
Coordinator
     │
     │ Authentication Token
     ▼
Worker
     │
     ├── Verify identity
     ├── Establish secure channel
     ├── Register hardware
     └── Register network capability
```

Security should be designed into the communication layer rather than added at the end.

---

# 15. Proposed CLI

## Coordinator

```bash
export FSSLK_PORT=27494
export FSSLK_MIN_INTERNET_SPEED=8000000000

fsslk --start
```

Alternatively:

```bash
fsslk start \
    --port 27494 \
    --min-bandwidth 8Gbps
```

## Worker

```bash
fsslk --join coordinator:27494
```

## Cluster status

```bash
fsslk status
```

Example:

```text
FSSLK Cluster

Nodes: 8

ID       GPU       BW       RTT      SCORE
------------------------------------------------
node01   GPU-A     10Gbps   4ms      98
node02   GPU-B      8Gbps   7ms      94
node03   GPU-C      2Gbps  71ms      53
node04   GPU-D     12Gbps   3ms      99
```

---

# 16. Environment Variables

The initial implementation can provide simple environment variables:

```bash
export FSSLK_PORT=27494
export FSSLK_MIN_INTERNET_SPEED=8000000000
```

Future versions may support:

```text
FSSLK_MIN_BANDWIDTH
FSSLK_MAX_RTT
FSSLK_MAX_PACKET_LOSS
FSSLK_COMPRESSION
FSSLK_QUANTIZATION
FSSLK_SCHEDULER
FSSLK_HEARTBEAT_INTERVAL
FSSLK_SECURITY_MODE
```

---

# 17. Proposed Repository Structure

```text
fsslk/
├── CMakeLists.txt
├── README.md
│
├── core/
│   ├── coordinator/
│   ├── worker/
│   ├── scheduler/
│   └── cluster/
│
├── network/
│   ├── transport/
│   ├── bandwidth/
│   ├── latency/
│   └── heartbeat/
│
├── tensor/
│   ├── transfer/
│   ├── compression/
│   ├── quantization/
│   └── cache/
│
├── pytorch/
│   ├── device/
│   ├── process_group/
│   ├── autograd/
│   └── bindings/
│
├── security/
│   ├── authentication/
│   ├── encryption/
│   └── authorization/
│
├── cli/
│   └── fsslk
│
└── tests/
    ├── unit/
    ├── integration/
    └── distributed/
```

---

# 18. Success Criteria

FSSLK development can be evaluated through progressive milestones.

### Level 1 — Connectivity

Multiple Internet-connected machines can join a cluster.

### Level 2 — Reliable Data Transfer

FSSLK can reliably transfer tensors between nodes.

### Level 3 — Distributed Execution

PyTorch workloads can execute across multiple machines.

### Level 4 — PyTorch Distributed Integration

FSSLK can operate as a PyTorch distributed backend.

### Level 5 — Adaptive Scheduling

Workload allocation responds to network and hardware conditions.

### Level 6 — WAN Communication Optimization

Compression, quantization, pipelining, caching, and transfer scheduling reduce communication overhead.

### Level 7 — Fault Tolerance

The cluster can respond to node failures and changing network conditions.

### Level 8 — Transparent Distributed Computing

Users can utilize heterogeneous Internet-connected computing resources without manually managing the underlying distributed system.

---

# 19. Final Vision

FSSLK is not simply intended to:

> "Run PyTorch on multiple computers."

The long-term vision is:

> **Turn heterogeneous computers connected through the Internet into a dynamic computing resource pool, and automatically schedule, transfer, synchronize, and optimize PyTorch workloads according to real-time hardware and network conditions.**

Conceptually:

```text
                       INTERNET
                           │
          ┌────────────────┼────────────────┐
          │                │                │
       Personal PC       Server           Cloud
          │                │                │
          └────────────────┼────────────────┘
                           │
                     ┌─────▼─────┐
                     │   FSSLK   │
                     │  Runtime  │
                     └─────┬─────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
      Scheduling       Transport       Optimization
          │                │                │
          └────────────────┼────────────────┘
                           │
                        PyTorch
                           │
                           ▼
                  Distributed AI Workload
```

## The Ultimate Goal

A user starts training on one machine:

```text
"Start training."
```

FSSLK then:

```text
Discover available nodes
        ↓
Measure network conditions
        ↓
Measure compute capabilities
        ↓
Select useful resources
        ↓
Partition the workload
        ↓
Optimize tensor transfers
        ↓
Synchronize when beneficial
        ↓
React to changing network conditions
        ↓
Recover from node failures
        ↓
Continue the workload
```

The user should not need to understand the underlying complexity.

### Final Objective

**FSSLK aims to make Internet-scale heterogeneous computing feel like a programmable, adaptive compute fabric for PyTorch.**