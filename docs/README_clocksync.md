# ChronoSync: Multi-Node Clock Synchronization

ChronoSync enables accurate cross-node timestamp correlation for distributed GPU workloads. It uses the Firefly clock synchronization algorithm to measure and correct clock offset and drift between nodes, so that traces from multiple machines can be merged and compared with consistent timing.

## How it works

ChronoSync uses a hub-and-spoke topology. One node (rank 0) acts as the reference clock and runs a probe server. One process per non-master node (the LOCAL_RANK=0 delegate) connects to the server and continuously exchanges UDP timing probes. A PI (proportional-integral) controller tracks both clock offset and drift rate, achieving ~2μs accuracy on direct ethernet and ~10μs on typical datacenter networks. The correction is published to a POSIX shared memory region that all profiled processes on the node can read. All other local processes block during initialization until the first valid offset is available, ensuring no uncorrected timestamps are recorded.

## Configuration

ChronoSync is configured via environment variables. Two modes are supported:

### With torchrun (automatic)

When running under torchrun, ChronoSync uses the environment variables that torchrun already provides. No additional configuration is needed:

- `GROUP_RANK` — node rank (0 = server, others = clients)
- `MASTER_ADDR` — address of the rank 0 node

```bash
torchrun --nnodes=2 --nproc-per-node=8 \
  --master-addr=192.0.2.1 --master-port=29500 \
  train.py
```

### Standalone

For non-torchrun workloads, set the equivalent environment variables:

- `RPDT_CLOCKSYNC_RANK` — this node's rank (0 = server)
- `RPDT_CLOCKSYNC_MASTER` — IP address of the rank 0 node (not needed on rank 0)

```bash
# Node 0 (server)
export RPDT_CLOCKSYNC_RANK=0
LD_PRELOAD=librpd_tracer.so ./my_workload

# Node 1 (client)
export RPDT_CLOCKSYNC_RANK=1
export RPDT_CLOCKSYNC_MASTER=192.0.2.1
LD_PRELOAD=librpd_tracer.so ./my_workload
```

### Optional

- `RPDT_CLOCKSYNC_PORT` — base port for clock sync (default: 29123). The server listens on this TCP port. UDP probe ports are assigned starting at base+1.

## Architecture

```
Node 0 (rank 0)                    Node N (rank N)
┌──────────────────┐               ┌──────────────────┐
│  Agent/rpdrun     │               │  Agent/rpdrun     │
│  ┌──────────────┐│               │  ┌──────────────┐│
│  │ ChronoSync   ││  UDP probes   │  │ ChronoSync   ││
│  │ Server       ││◄─────────────►│  │ Client       ││
│  │ (offset = 0) ││               │  │ (computes    ││
│  └──────┬───────┘│               │  │  offset)     ││
│         │ shm     │               │  └──────┬───────┘│
│  ┌──────┴───────┐│               │  ┌──────┴───────┐│
│  │ Workers      ││               │  │ Workers      ││
│  │ (read offset)││               │  │ (read offset)││
│  └──────────────┘│               │  └──────────────┘│
└──────────────────┘               └──────────────────┘
```

### Role assignment

| Condition | Role |
|-----------|------|
| `GLOBAL_RANK == 0` | Clock sync server + logging receiver |
| `LOCAL_RANK == 0` (non-master) | Clock sync client + logging sender |
| All other ranks | Attach shared memory, block until synced |

### Per-node singleton

On each node, `DbResource` (SQLite-based lock) ensures only one process runs the clock sync service. The first process to acquire the lock becomes the delegate; others attach to its shared memory segment and block until a valid offset is published (`SvcState.sequence > 0`).

### Shared memory

The delegate creates a POSIX shared memory segment containing a `SvcState` structure protected by a seqlock. The hot-path read (`clocktime_ns()`) costs a single atomic load + branch when no sync is active (`sequence == 0`).

## Worker blocking

Non-delegate processes block during `init()` with a 60-second timeout. If the shared memory segment or a valid offset is not available within the timeout, a warning is printed to stderr and the process proceeds with uncorrected timestamps.

## Network requirements

- Nodes must be able to reach rank 0 via TCP on the clock sync port (default 29123).
- UDP ports starting at base+1 are used for probe exchanges (one per client).
- Nodes can join at different times; the server accepts connections as they arrive.

## Limitations

- Clock correction assumes symmetric network paths. Persistent path asymmetry introduces an irreducible bias (typically < 20μs on datacenter networks).
- The PI controller's frequency estimate is clamped at 500 ppm to prevent windup from noisy measurements.
- Clock correction degrades after the sync process exits. For long-running workloads, keep the sync process active for the duration of the trace.

## Testing

Unit tests are in `rpd_tracer/tests/`:

```bash
cd rpd_tracer/tests && make test
```

The `test_convergence` suite uses synthetic clocks and network models to verify convergence properties without requiring multiple physical nodes.
