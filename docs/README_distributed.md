# Distributed Profiling with rpd_tracer

rpd_tracer supports multi-node GPU profiling with cross-node clock synchronization and centralized trace collection. All nodes write timestamps aligned to a common reference clock, so traces from different machines can be merged and compared directly.

## Architecture

Each node has a **delegate** process that runs infrastructure services (clock sync, log aggregation) on behalf of all local GPU workers. The delegate outlives the workers, ensuring services remain available until all nodes have finished.

```
Node 0                               Node N
┌─────────────────────┐              ┌─────────────────────┐
│  Delegate (rpdrun)  │              │  Delegate (rpdrun)  │
│  ┌────────────────┐ │              │  ┌────────────────┐ │
│  │ Clock sync     │ │  UDP probes  │  │ Clock sync     │ │
│  │ server         │◄├──────────────┤► │ client         │ │
│  │ Log aggregation│ │              │  │ Log aggregation│ │
│  │ receiver       │◄├──────────────┤► │ sender         │ │
│  └───────┬────────┘ │              │  └───────┬────────┘ │
│     shm  │          │              │     shm  │          │
│  ┌───────┴────────┐ │              │  ┌───────┴────────┐ │
│  │ GPU Workers    │ │              │  │ GPU Workers    │ │
│  │ (read offset,  │ │              │  │ (read offset,  │ │
│  │  write trace)  │ │              │  │  write trace)  │ │
│  └────────────────┘ │              │  └────────────────┘ │
└─────────────────────┘              └─────────────────────┘
```

### Roles

| Condition | Role |
|-----------|------|
| Node 0 | Clock sync server, log aggregation receiver, master trace file |
| Node > 0 | Clock sync client, log aggregation sender |
| GPU workers | Attach shared memory, read clock offset, write trace events |

### Clock synchronization

The rank 0 node is the reference clock. All other nodes measure their clock offset from rank 0 using a PI (proportional-integral) controller driven by UDP probe exchanges. The correction is shared to all local GPU workers who can apply it with minimal overhead (one atomic load per timestamp).

Accuracy: ~2μs on direct ethernet, ~10μs on typical datacenter networks.

## Using runTracer.sh

`runTracer.sh` is the primary interface for standalone (non-torchrun) distributed profiling.

### Single node (unchanged from single-node usage)

```bash
runTracer.sh -o trace.rpd python train.py
```

### Multi-node

On each node, pass `--rank` and `--master`:

```bash
# Node 0 (rank 0 — reference clock and trace aggregator)
runTracer.sh -o trace.rpd --rank 0 python train.py

# Node 1
runTracer.sh -o trace.rpd --rank 1 --master 192.0.2.1 python train.py

# Node 2
runTracer.sh -o trace.rpd --rank 2 --master 192.0.2.1 python train.py
```

The `--master` flag is the IP address of the rank 0 node. Rank 0 does not need `--master`.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o FILE` | trace.rpd | Output trace file |
| `--rank N` | (none) | Node rank. Enables multi-node mode. 0 = server. |
| `--master ADDR` | (none) | IP of the rank 0 node (required for rank > 0) |
| `--exit-delay N` | 30 | Seconds rank 0 waits after child exits for remote nodes to flush |
| `--load` | (off) | Load-only mode: sets AUTOSTART=0, DELAYINIT=1 for programmatic control |

### Load-only mode

For applications that control the recording window programmatically via `rpdstart()`/`rpdstop()`:

```bash
runTracer.sh --load --rank 0 python train.py
```

`loadTracer.sh` is a shorthand for `runTracer.sh --load`.

### Controlling the recording window

By default, rpd_tracer records the entire process lifetime. To record only specific sections:

- **Environment**: set `RPDT_AUTOSTART=0` before running. Call `rpdstart()`/`rpdstop()` from your code.
- **PyTorch**: use `torch.profiler.rpd_profile()` as a context manager. It calls `rpdstart()`/`rpdstop()` internally.

Both approaches work with multi-node profiling. When `RPDT_AUTOSTART=0`, workers do not record until explicitly started. The delegate's infrastructure services (clock sync, log aggregation) run regardless of the recording window.

## How rpdrun works

When `--rank` is specified, `runTracer.sh` delegates to `rpdrun`, a thin wrapper that loads `librpd_tracer.so` and becomes the per-node delegate before spawning the user command. The delegate runs clock sync and log aggregation services, outliving the child process to ensure remote nodes can flush. GPU workers attach to the delegate's shared memory rather than starting their own services.

## Network requirements

- Rank 0 must be reachable via TCP on the clock sync port (default 29123)
- UDP ports starting at clock sync port + 1 are assigned per client
- Nodes can join at any time; the server accepts connections as they arrive

## Environment variables

| Variable | Description |
|----------|-------------|
| `RPDT_CLOCKSYNC_RANK` | Node rank (set by runTracer.sh from `--rank`) |
| `RPDT_CLOCKSYNC_MASTER` | Rank 0 address (set by runTracer.sh from `--master`) |
| `RPDT_CLOCKSYNC_PORT` | Clock sync port (default 29123) |
| `RPDT_LOGAGG_PORT` | Log aggregation port (default 29223) |
| `RPDT_FILENAME` | Output trace file path |
| `RPDT_AUTOSTART` | Start recording immediately (default 1) |
| `RPDT_QUIET` | Suppress informational output (default 0) |
