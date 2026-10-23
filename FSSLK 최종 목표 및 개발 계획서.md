# FSSLK — Final Goal & Development Plan

## 1. 프로젝트 개요

**FSSLK**는 인터넷으로 연결된 여러 컴퓨터의 CPU/GPU 자원을 하나의 분산 연산 환경처럼 활용할 수 있도록 만드는 **PyTorch 기반 WAN-aware 분산 학습/연산 런타임**이다.

기존의 PyTorch RPC나 일반적인 Distributed Data Parallel(DDP)이 주로 안정적인 데이터센터/LAN 환경을 전제로 하는 것과 달리, FSSLK는 다음과 같은 환경을 주요 대상으로 한다.

- 서로 다른 인터넷 환경에 존재하는 컴퓨터
- 서로 다른 CPU/GPU 성능
- 서로 다른 네트워크 대역폭
- 서로 다른 RTT와 packet loss
- 동적으로 참여하거나 이탈하는 노드
- 가정용 PC, 개인 서버, 클라우드 서버 등이 혼합된 heterogeneous 환경

FSSLK의 핵심 목표는 사용자가 개별 노드의 복잡한 통신 및 작업 분배를 직접 관리하지 않아도 **런타임이 네트워크와 연산 능력을 측정하고 적절한 분산 전략을 자동으로 선택하도록 만드는 것**이다.

---

# 2. 최종 목표

최종적으로 다음과 같은 사용 경험을 목표로 한다.

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

노드가 연결되면 FSSLK가 자동으로:

```text
Node Discovery
      ↓
Authentication
      ↓
Network Benchmark
      ↓
Hardware Benchmark
      ↓
Capability Registration
      ↓
Distributed Scheduler
      ↓
PyTorch Integration
      ↓
Automatic Execution
```

을 수행한다.

사용자는 가능한 경우 기존 PyTorch 코드의 변경을 최소화한다.

최종적인 이상적인 사용 형태는 다음과 같다.

```python
import torch
import fsslk

fsslk.init()

model = MyModel()
model = fsslk.distribute(model)

output = model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

즉, 사용자가 직접 다음과 같은 것을 관리하지 않는 것을 목표로 한다.

```text
어떤 노드에 작업을 보낼 것인가?
얼마나 데이터를 보낼 것인가?
언제 synchronization할 것인가?
어떤 노드를 제외할 것인가?
gradient를 압축할 것인가?
어떤 통신 경로가 가장 빠른가?
```

이러한 판단을 FSSLK runtime이 담당한다.

---

# 3. 핵심 설계 철학

## 3.1 RPC를 대체하지 않는다

FSSLK는 단순한 RPC 재구현 프로젝트가 아니다.

RPC의 장점인:

- 원격 함수 실행
- 비동기 실행
- Remote object
- 분산 autograd

등은 필요에 따라 활용한다.

FSSLK가 추가하려는 핵심 영역은:

```text
RPC
 └─ Remote execution

FSSLK
 ├─ Network awareness
 ├─ Hardware awareness
 ├─ Dynamic scheduling
 ├─ WAN optimization
 ├─ Tensor communication optimization
 ├─ Node management
 └─ Failure handling
```

이다.

---

# 4. 전체 아키텍처

```text
                         PyTorch Application
                                │
                                ▼
                     ┌─────────────────────┐
                     │   FSSLK PyTorch API │
                     └──────────┬──────────┘
                                │
             ┌──────────────────┴──────────────────┐
             │                                     │
             ▼                                     ▼
    Custom Device Backend                 Distributed Backend
       / PrivateUse1                         / ProcessGroup
             │                                     │
             └──────────────────┬──────────────────┘
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
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
          Node A             Node B             Node C
          CPU/GPU            CPU/GPU            CPU/GPU
```

---

# 5. 단계별 개발 계획

## Phase 0 — 프로토타입

목표:

**"인터넷을 통해 여러 노드가 연결되고 작업을 실행할 수 있다."**

구현:

- Coordinator
- Worker
- node ID
- token authentication
- TCP/QUIC 기반 통신
- heartbeat
- 기본 RPC
- 기본 tensor transfer

예:

```bash
fsslk --start
```

```bash
fsslk --join 192.168.1.10:27494
```

결과:

```text
[FSSLK] Coordinator started
[FSSLK] Node joined: node-01
[FSSLK] Node joined: node-02
[FSSLK] Cluster size: 3
```

---

# 6. Phase 1 — Network Awareness

FSSLK의 첫 번째 핵심 기능.

각 노드에서 주기적으로:

```text
Bandwidth
RTT
Jitter
Packet Loss
Connection Stability
```

를 측정한다.

예:

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

이 정보를 scheduler가 사용한다.

단순한:

```text
"8 Gbps 이상이면 참여"
```

에서 발전하여:

```text
Network Score =
f(bandwidth, RTT, jitter, loss, stability)
```

형태의 score를 만든다.

---

# 7. Phase 2 — Hardware Awareness

각 Worker의 연산 능력을 측정한다.

수집 정보:

```text
CPU cores
CPU throughput
GPU type
GPU memory
GPU throughput
RAM
available VRAM
available RAM
```

가능하면 짧은 benchmark를 수행하여 실제 성능을 추정한다.

예:

```text
Node A
GPU: X
Compute: 100 TFLOPS
VRAM: 24 GB

Node B
GPU: Y
Compute: 35 TFLOPS
VRAM: 12 GB
```

이를 scheduler에 반영한다.

---

# 8. Phase 3 — Adaptive Scheduler

FSSLK의 핵심 기술.

각 노드의:

```text
Compute Capability
+
Network Capability
+
Current Load
+
Memory
+
Communication Cost
```

를 이용하여 작업을 분배한다.

예:

```text
Total workload = 100

Node A → 50
Node B → 10
Node C → 40
```

고정된 `world_size` 기반 분배가 아니라 runtime 상태에 따라 동적으로 변경하는 것을 목표로 한다.

---

# 9. Phase 4 — Tensor Communication Engine

대규모 tensor를 인터넷으로 전송할 때의 비용을 줄인다.

지원 목표:

```text
Tensor chunking
Parallel transfer
Compression
Quantization
Caching
Prefetching
Pipelining
```

예:

```text
Tensor
  │
  ├─ Chunk 1 ─┐
  ├─ Chunk 2 ─┼── Network
  ├─ Chunk 3 ─┤
  └─ Chunk 4 ─┘
```

네트워크 상태가 나쁘면:

```text
FP32
 ↓
FP16 / BF16 / INT8
 ↓
Transfer
 ↓
Reconstruction
```

과 같은 전략을 선택할 수 있도록 설계한다.

단, 정확도와 성능의 trade-off를 측정하여 **정확도 손실을 최소화하는 방향**으로 구현한다.

---

# 10. Phase 5 — PyTorch Distributed Backend

FSSLK를 PyTorch의 distributed infrastructure와 연결한다.

주요 목표:

```text
FSSLK ProcessGroup
```

을 구현하고 다음과 같은 collective를 지원한다.

```text
all_reduce
all_gather
reduce_scatter
broadcast
send
recv
```

초기에는 단순한 구현으로 시작하고 이후:

```text
adaptive all-reduce
compressed all-reduce
hierarchical communication
WAN-aware synchronization
```

등으로 발전시킨다.

---

# 11. Phase 6 — PyTorch Custom Device Integration

필요한 경우 PyTorch의 custom device/backend 확장 기능을 사용한다.

목표:

```text
torch.device("fsslk")
```

또는 PyTorch에서 허용되는 custom backend/device 메커니즘을 활용하여 FSSLK가 제공하는 실행 환경을 PyTorch에 자연스럽게 연결한다.

중요한 원칙은:

> Custom device와 distributed communication을 하나의 개념으로 섞지 않는다.

Device backend는 **연산/메모리 abstraction**을 담당하고,

Distributed backend는 **노드 간 통신**을 담당한다.

---

# 12. Phase 7 — Automatic Optimization

최종적으로 FSSLK runtime이 다음을 자동 결정한다.

```text
┌─────────────────────────────┐
│       Current Cluster       │
├─────────────────────────────┤
│ Network                     │
│ Hardware                    │
│ Memory                      │
│ Load                        │
│ Latency                     │
│ Packet Loss                 │
└──────────────┬──────────────┘
               │
               ▼
        FSSLK Optimizer
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
   Placement  Transfer  Sync
   Strategy   Strategy  Strategy
```

예를 들어:

```text
Node A
GPU 매우 빠름
Network 매우 빠름
→ 큰 workload

Node B
GPU 느림
Network 빠름
→ 작은 workload / communication 역할

Node C
GPU 빠름
Network 느림
→ local computation 비중 증가
```

와 같은 판단을 runtime이 수행한다.

---

# 13. Phase 8 — Fault Tolerance

인터넷 기반 분산학습에서 매우 중요하다.

다음 상황을 처리한다.

```text
Node disconnect
Node timeout
Bandwidth degradation
Packet loss increase
GPU failure
Process crash
Network partition
```

예:

```text
Node B
   ↓
connection lost
   ↓
FSSLK detects failure
   ↓
remaining nodes redistribute workload
   ↓
training continues
```

목표는 **하나의 개인 PC가 꺼졌다고 전체 학습이 즉시 실패하지 않는 구조**다.

---

# 14. Phase 9 — Security

인터넷에 노드를 직접 연결하기 때문에 필수적으로 구현한다.

필요 기능:

```text
Authentication
Authorization
Encrypted transport
Node identity
Token rotation
Replay protection
Rate limiting
Resource limits
```

Coordinator가 발급한 token을 이용해 Worker가 클러스터에 참여하는 구조를 고려한다.

예:

```bash
fsslk --join coordinator.example:27494
```

```text
Token:
***************
```

인증 후:

```text
Node identity
Hardware capability
Network capability
```

를 등록한다.

---

# 15. 최종 CLI

최종적으로 다음과 같은 UX를 목표로 한다.

## Coordinator

```bash
export FSSLK_PORT=27494
export FSSLK_MIN_INTERNET_SPEED=8000000000

fsslk --start
```

또는:

```bash
fsslk start \
    --port 27494 \
    --min-bandwidth 8Gbps
```

## Worker

```bash
fsslk --join coordinator:27494
```

## 상태 확인

```bash
fsslk status
```

출력 예:

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

# 16. 환경변수

초기에는 단순한 환경변수를 지원한다.

```bash
export FSSLK_PORT=27494
export FSSLK_MIN_INTERNET_SPEED=8000000000
```

이후에는:

```bash
FSSLK_MIN_BANDWIDTH
FSSLK_MAX_RTT
FSSLK_MAX_PACKET_LOSS
FSSLK_COMPRESSION
FSSLK_QUANTIZATION
FSSLK_SCHEDULER
FSSLK_HEARTBEAT_INTERVAL
FSSLK_SECURITY_MODE
```

등을 지원한다.

---

# 17. 프로젝트 디렉터리 예시

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

# 18. 성공 기준

FSSLK의 성공을 다음 단계로 정의한다.

### Level 1

인터넷으로 여러 노드가 연결된다.

### Level 2

노드 간 tensor를 안정적으로 전송한다.

### Level 3

PyTorch tensor 연산을 여러 노드에서 실행한다.

### Level 4

PyTorch distributed backend로 사용할 수 있다.

### Level 5

네트워크 상태에 따라 workload를 자동 조절한다.

### Level 6

tensor compression/quantization/pipelining으로 WAN 통신량을 줄인다.

### Level 7

노드 장애와 네트워크 변화에 자동 대응한다.

### Level 8

사용자가 복잡한 distributed 설정을 거의 하지 않고도 인터넷상의 heterogeneous computing resource를 활용할 수 있다.

---

# 19. 최종 비전

FSSLK의 최종 목표는 단순히

> "여러 컴퓨터에서 PyTorch를 실행한다."

가 아니다.

궁극적으로는:

> **인터넷에 연결된 서로 다른 컴퓨터들을 하나의 동적인 계산 자원 풀(pool)처럼 만들고, PyTorch workload를 네트워크와 하드웨어 상황에 맞춰 자동으로 배치·전송·동기화하는 runtime을 구축한다.**

것을 목표로 한다.

```text
                    INTERNET
                        │
        ┌───────────────┼───────────────┐
        │               │               │
      PC/GPU          Server          Cloud
        │               │               │
        └───────────────┼───────────────┘
                        │
                  ┌─────▼─────┐
                  │   FSSLK   │
                  │  Runtime  │
                  └─────┬─────┘
                        │
        ┌───────────────┼────────────────┐
        │               │                │
     Scheduling      Transport       Optimization
        │               │                │
        └───────────────┼────────────────┘
                        │
                     PyTorch
                        │
                        ▼
                 Distributed AI
```

## 최종적으로 만들고 싶은 것

```text
"컴퓨터 A에서 학습을 시작한다."

        ↓

FSSLK

        ↓

"인터넷에 연결된 컴퓨터 중
현재 가장 효율적인 자원을 자동 선택한다."

        ↓

"필요한 tensor만 전송한다."

        ↓

"네트워크가 느려지면 전략을 변경한다."

        ↓

"노드가 사라지면 workload를 재배치한다."

        ↓

"사용자는 이 과정을 거의 신경 쓰지 않는다."
```

이것이 FSSLK의 최종 목표다.