# Using rpd_tracer with torchrun

rpd_tracer integrates with torchrun for multi-node distributed training profiling. The `--profile-url` flag enables profiling with automatic clock synchronization across nodes.

## Quick start

```bash
# Node 0
torchrun --nnodes=2 --nproc-per-node=8 \
  --master-addr=192.0.2.1 --master-port=29500 \
  --profile-url /output/trace.rpd \
  train.py

# Node 1
torchrun --nnodes=2 --nproc-per-node=8 \
  --master-addr=192.0.2.1 --master-port=29500 \
  --profile-url /tmp/trace.rpd \
  train.py
```

`--profile-url` must be specified on every node. The trace file on node 0 is the final aggregated output. Files on other nodes are used for local coordination only.

## How it works

When `--profile-url` is set, the torchrun agent loads `librpd_tracer.so` after rendezvous and becomes the per-node delegate for clock sync and log aggregation. Workers are profiled automatically. The agent outlives the workers, keeping services available until all nodes have finished.

Node 0 is the reference clock. All other nodes synchronize to it, achieving ~2μs accuracy on direct ethernet. GPU workers apply the clock correction transparently — no changes to training code are needed.

## Recording window

By default (`RPDT_AUTOSTART=1`), workers record the entire process lifetime. To control the recording window:

### Full trace (default)

```bash
torchrun --profile-url trace.rpd train.py
```

### Windowed trace with rpd_profile

Set `RPDT_AUTOSTART=0` and use `rpd_profile` in your training script:

```bash
RPDT_AUTOSTART=0 torchrun --profile-url trace.rpd train.py
```

```python
# train.py
from torch.profiler import rpd_profile

with rpd_profile(record_shapes=True) as p:
    for step in range(100):
        train_step()
        p.step()
```

`rpd_profile` calls `rpdstart()`/`rpdstop()` internally. Clock sync and log aggregation run regardless of the recording window.

## Single node

`--profile-url` works for single-node multi-GPU as well:

```bash
torchrun --nproc-per-node=8 --profile-url trace.rpd train.py
```

## Network requirements

- Clock sync uses TCP port 29123 (configurable via `RPDT_CLOCKSYNC_PORT`)
- UDP ports starting at 29124 are assigned per client node
- Log aggregation uses TCP port 29223 (configurable via `RPDT_LOGAGG_PORT`)
- These are separate from torchrun's `--master-port` (used for rendezvous)

## Environment variables

| Variable | Description |
|----------|-------------|
| `RPDT_AUTOSTART` | Set to `0` before torchrun to use `rpd_profile` for windowed recording |
| `RPDT_CLOCKSYNC_PORT` | Clock sync port override (default 29123) |
| `RPDT_LOGAGG_PORT` | Log aggregation port override (default 29223) |

All other profiling environment variables are managed by torchrun automatically.

## Standalone alternative

For distributed profiling without torchrun, use `runTracer.sh --rank`. See [README_distributed.md](README_distributed.md).
