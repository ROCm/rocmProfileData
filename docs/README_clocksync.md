# ChronoSync: Multi-Node Clock Synchronization

ChronoSync enables accurate cross-node timestamp correlation for distributed GPU workloads. It uses the Firefly clock synchronization algorithm to measure and correct clock offset and drift between nodes, so that traces from multiple machines can be merged and compared with consistent timing.

## How it works

When enabled, one process per node runs the Firefly protocol — exchanging UDP timing probes with peer nodes to compute clock offset and drift via linear regression. The correction is applied to all timestamps (CPU-side API calls and GPU activity from roctracer/rocprofiler-sdk/cupti) via a shared memory region that all profiled processes on the node can read.

## Configuration

Two environment variables control ChronoSync:

### `RPDT_CLOCKSYNC_IP`

Path to a configuration file listing all participating nodes. Each line has the format:

```
<ip_address>,rank=<rank_number>
```

Example config file for a two-node setup:

```
192.0.2.1,rank=0
192.0.2.2,rank=1
```

### `RPDT_CLOCKSYNC_RANK`

The rank of the current node. Must match one of the ranks in the config file.

## Usage

1. Create a config file listing all nodes and their ranks.

2. On each node, set the environment variables and run the workload with rpd_tracer:

```bash
# Node 0
export RPDT_CLOCKSYNC_IP=/path/to/sync_config.txt
export RPDT_CLOCKSYNC_RANK=0
LD_PRELOAD=librpd_tracer.so ./my_workload

# Node 1
export RPDT_CLOCKSYNC_IP=/path/to/sync_config.txt
export RPDT_CLOCKSYNC_RANK=1
LD_PRELOAD=librpd_tracer.so ./my_workload
```

3. All processes on each node that write to the same RPD file automatically share the clock correction via POSIX shared memory. No additional configuration is needed for multi-process workloads.

## Network requirements

- Nodes must be able to reach each other via UDP and TCP on port `12345 + rank_a + rank_b` for each pair of nodes.
- The Firefly protocol uses a TCP handshake for initial synchronization, then UDP probes for ongoing measurement.

## Limitations

- Clock correction degrades over time after the sync process exits. For long-running sequential workloads, keep the sync process active for the duration of the trace.
- The drift rate is clamped at 500 ppm. If the measured drift exceeds this threshold (indicating noisy measurements), drift correction is disabled and only the offset is applied.
