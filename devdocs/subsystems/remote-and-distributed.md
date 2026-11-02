# remote/ and the distributed stack

**Two unrelated things are called "remote". Never confuse them:**

| Name | What | Mechanism |
|---|---|---|
| `remote/` (`librpd_remote.so`, `rpdRemote`) | single-node **recording-window trigger** | SIGUSR1/SIGUSR2 handlers → dlopen tracer → `rpdstart/rpdstop/rpdflush` |
| `rpd_tracer/RemoteDataSource.{h,cpp}` + `NetWriterBackend` | multi-node **record aggregation receiver** | TCP into the shared `.rpd` on node 0 |

## remote/ — the signal trigger (Scope: whole dir, 4 tracked files)

`Remote.cpp:18-23`: library constructor installs SIGUSR1 → `startTracing`
(`dlopen("librpd_tracer.so")` + `dlsym("rpdstart")`, nested-start refcount)
and SIGUSR2 → stop+flush (`Remote.cpp:25-61`). CLI:
`rpdRemote <start|stop> <pid>` = `kill -s SIGUSR1|SIGUSR2 <pid>`
(`rpdRemote.sh:24-40`). Tracked-file usage:

```
python3 -m rocpd.schema --create trace.rpd     # file must pre-exist
RPDT_FILENAME=trace.rpd LD_PRELOAD=librpd_remote.so python3 app.py &
rpdRemote start <pid> ; ... ; rpdRemote stop <pid>
```

Caveats (FEATURES.md:72-88): tracer can't be unloaded once loaded; output file
fixed per process lifetime; `dlopen/dlsym` inside a signal handler is not
async-signal-safe (known); stop-path refcount leaks if the start dlopen
failed; no unload by design (`Remote.cpp:62-67`). Built by both the legacy
Makefile and CMake (`remote/CMakeLists.txt:4-21`; installs script renamed
`rpdRemote`). Not referenced by the distributed docs.

## Distributed tracing (implemented inside rpd_tracer)

User-facing docs: `docs/README_distributed.md`, `docs/README_clocksync.md`,
`docs/README_torchrun.md`. Landed 2026-09 in the 12-PR distdev series
(#122–#133) — actively changing.

```
 rank 0: runTracer.sh --rank 0 --master H → rpdrun (delegate)
           dlopens librpd_tracer.so as services host (RPDT_AGENT=1,
           GPU datasources excluded, RPDT_AUTOSTART=0 — rpdrun.c:58-63),
           then forks the app with LD_PRELOAD and outlives it (--exit-delay)
 rank N: same, but NetWriterBackend replaces every table's sqlite backend
           (isRemoteNode() gate Utility.h:135-145)
           │ TCP wire protocol (NetWriterBackend.h:14-47):
           │   32-byte tag + directWrite handshake, then
           │   [size|idOffset|nodeId|rowCount|rows]; flush = rowCount 0
           ▼
 RemoteDataSource on node 0 (Logger.cpp:211-216 adds it iff
 RPDT_LOGAGG_PORT set AND RPDT_NODE_ID==0; delegate elected via
 DbResource("remote_active") + port bind, RemoteDataSource.cpp:189-227):
   accept thread + one recv thread per connection + 9 writer threads
   (one per table, own connection/temp tables) draining into the normal
   *Table writers
```

**Id-space layout** (must not overflow; documented
`RemoteDataSource.cpp:24-49`, applied at the receiver): row id bits 0–30,
annotation flag bit 31, sessionId bits 32–47 (per-process offset from
`Storage.cpp:36-45`), nodeId bits 48–62. Additionally pid/tid +=
`nodeId*pid_stride` (10,000,000) and gpuId += `nodeId*gpu_stride` (1000),
strides stored in `rocpd_metadata`; de-striding views `napi`/`nop` undo it
(`utilitySchema.cmd:27-28`).

**Clock sync** = ChronoSyncDataSource + Firefly (see
[concepts/timestamps-and-clocksync.md](../concepts/timestamps-and-clocksync.md)).
torchrun mode: `torchrun --profile-url <path>` auto-configures everything from
`GROUP_RANK`/`MASTER_ADDR` (`docs/README_torchrun.md`).

Failure model: if the node-0 delegate dies, in-flight data is lost — accepted
for a profiler (`RemoteDataSource.h:21-23`); flush waits are bounded to 5s and
skip stopped channels (`RemoteDataSource.cpp:327-365`).

## How to test multi-node locally

Two ranks on one box: `runTracer.sh --rank 0 --master localhost app.py` in one
shell, `--rank 1` in another; check `rocpd_metadata` for `remote_delegate` /
`clocksync_shm` rows and query `napi`/`nop` views. Clocksync units:
`ctest` with `-DRPD_BUILD_TESTS=ON` (`test_seqlock`, `test_measurement`,
`test_regression`, `test_convergence`).
