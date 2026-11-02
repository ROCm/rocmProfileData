# Injection & interposition

How code gets into the traced process, and every symbol boundary involved.

## Primary: LD_PRELOAD + ELF constructor

`runTracer.sh` is just env + `LD_PRELOAD=librpd_tracer.so "$@"`
(`rpd_tracer/runTracer.sh:69`; `loadTracer.sh` = same with
`RPDT_AUTOSTART=0 RPDT_DELAYINIT=1` to attach without tracing). On load,
`__attribute__((constructor)) rpdTracerInit()` (`rpd_capi.cpp:90`) builds the
Logger. There is **no link against librocprofiler-register anywhere.**

**Exit ordering trick** (`rpd_capi.cpp:124-146`): the tracer interposes
`atexit`/`__cxa_atexit`, forwards each user registration, then re-appends its
own finalize *after* it — so at LIFO exit time `rpdFinalize()` runs **before
any user atexit handler**, while sqlite and other libs are still alive. A
destructor and the idempotence guard (`Logger.cpp:263-290`) back it up.

## Variants

| Path | Mechanism |
|---|---|
| Embedded | link/dlopen `librpd_embedded.so`; call `rpdstart/rpdstop/rpdflush/...` (`rpd_tracer.h:9-16`); autostart forced off (`rpd_capi.cpp:148-156`); `docs/README_embedded.md` |
| Python | `rpdTracerControl` sets `RPDT_AUTOSTART=0` + `RPDT_FILENAME`, `CDLL(find_library("rpd_tracer"))` (`rpdTracerControl.py:31-36`); `RPDT_LOADED` env lets it detect an already-preloaded tracer (`:82`) |
| Signal trigger | `LD_PRELOAD=librpd_remote.so` + `rpdRemote start <pid>` → SIGUSR1 → dlopen tracer on demand (`remote/Remote.cpp`) |
| Multi-node | `rpdrun.c` dlopens the tracer into itself as a delegate (agent env, GPU sources excluded), restores the user's env, then execs the app with `LD_PRELOAD` (`rpdrun.c:58-83`) |

## Symbol boundaries the tracer OWNS (LD_PRELOAD interpositions)

1. `atexit` / `__cxa_atexit` — above.
2. **Marker shims**: exports `roctxMarkA/roctxRangePushA/roctxRangePop` +
   NVTX trio via the linker version script `rpd_tracer/marker_shim.ver`, so
   apps linking markers need neither libroctx64 nor libcupti; hits land in
   `rocpd_api` (Roctx/NvtxDataSources). rlog apps route through `librlog.so`
   instead (or the tracer's roctx shims when rlog forwards).
3. GPU provider hooks are attached dynamically (dlsym'd CLR/SDK/roctracer/CUPTI
   entry points) rather than by symbol interposition.

## rocprofiler-register interplay (ROCm 7.x trap)

`RocprofDataSource.cpp:1171-1188` exports `rocprofiler_configure` — that's the
SDK tool handshake, *not* injection. Side effect: rocprofiler-register scans
loaded libs for that symbol and considers registration satisfied, skipping the
legacy `HSA_TOOLS_LIB`/`OnLoad` path — which would starve RTL. So
`RtlDataSource.cpp:163-178` force-sets `HSA_TOOLS_LIB` to the tracer itself
and `HSA_TOOLS_ROCPROFILER_V1_TOOLS=1`. Any future HSA-tools feature must
respect the same dance.

## RTL standalone (different direction)

Standalone rocm-trace-lite *is* the HSA tools library: `HSA_TOOLS_LIB=librtl.so`
→ `OnLoad()` swaps HSA API-table entries (`hsa_queue_create`,
`hsa_executable_freeze`); `RTL_MODE=hip` additionally LD_PRELOADs HIP wrappers
(`dlsym(RTLD_NEXT)` with an RTLD_NOLOAD fallback for spawn workers,
`src/hip_api_intercept.cpp:147-187`).

## fork/exec rules

Supported model is **fork+exec under LD_PRELOAD**; each process initializes
normally, per-PID session offsets make them share one file
(`Storage.cpp:36-45`). Plain `fork()` without `exec()` is unsupported:
per-table worker threads don't exist in the child (no `pthread_atfork`
anywhere), and pid/tid caches are thread-local re-derived values
(`Utility.h:26-36`).
