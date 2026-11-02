# Timestamps & clock sync

## The single clock rule

Every local timestamp originates from **`clocktime_ns()`**
(`rpd_tracer/Utility.h:93-98`) = `CLOCK_MONOTONIC` + the Firefly correction
(offset + drift·elapsed, read via seqlock). **Never call `clock_gettime`
directly** for a trace row — that bypasses multi-node alignment. ~80 call
sites go through this one function (`Utility.h:51-52`).

Provider records that arrive already stamped in a foreign clock domain must
pass through **`adjust_external_ts()`** (`Utility.h:89-91`). Backends differ
in how they handle this — a real correctness surface:

| Source | Clock handling |
|---|---|
| Roctracer, Rocprof | `adjust_external_ts` on SDK/roctracer timestamps |
| Clr, Rtl | provider ns used **unmodified** (`ClrDataSource.cpp:146-147`, `RtlDataSource.cpp:70-71`) — assumed same monotonic domain |
| Cupti | CUPTI uses CLOCK_REALTIME; converts by sampling both clocks at buffer-completion (approximate, self-described "hopefully doing no additional damage", `CuptiDataSource.cpp:652-658`) |
| RTL standalone | host events use CLOCK_MONOTONIC (comment asserts AMD TSC domain matches, `trace_db.cpp:18-28`); GPU times from `hsa_amd_profiling_get_dispatch_time`; `end<=start` dropped (`hsa_intercept.cpp:255-256`) |

DB stores int64 ns, monotonic domain — only deltas meaningful; GPU durations
can be negative (`rpd_viewer/RPD_INFO.md` §1). Viewers divide by 1000 at the
edge (µs for Chrome/Perfetto; `top` view reports µs).

## Correlation ids

- GPU work → launching API: provider correlation id lands in
  `rocpd_op.api_id`; bridge table `rocpd_api_ops`. Roctracer/Rocprof wire this;
  **RTL does NOT persist correlation** (Phase-2 unwired,
  `rocm-trace-lite/src/hip_api_intercept.cpp:26-33`) — RTL-embedded traces lack
  api↔op links.
- App-level ranges: `Logger::nextAnnotationId()`, occupies the bit-31+ region
  (`Storage.h:61`) so annotation ids never collide with provider api ids.
- Id-space bit layout (row id / annotation / session<<32 / node<<48) is
  documented `rpd_tracer/RemoteDataSource.cpp:24-49`; must not overflow.

## Multi-node clock sync — Firefly / ChronoSync

Enabled only if `RPDT_CLOCKSYNC_RANK` set (`Logger.cpp:208-209`). User-facing
detail: `docs/README_clocksync.md`, `docs/README_distributed.md`.

Mechanism (`rpd_tracer/Firefly.{h,cpp}`, `ChronoSyncDataSource.cpp`):
- Hub-and-spoke **UDP** probes, rank 0 = server (offset 0), rank N>0 = client
  running a **PI controller** (Kp=0.1, Ki=0.02, anti-windup ±500ppm,
  `Firefly.h:43-44`) over median-of-window + least-squares-drift samples.
- **Software timestamps only** — `SOF_TIMESTAMPING_*_SOFTWARE`; `hw_now()` is
  just `clock_gettime(CLOCK_MONOTONIC)` despite the name. No PTP/hardware
  timestamping. Accuracy ~2µs direct ethernet, ~10µs datacenter.
- Per-node single service owner elected via `DbResource("chronosync_active")`;
  delegate builds a POSIX-shm `SvcState` {seqlock, referenceTime, offset,
  drift} and publishes its name in `rocpd_metadata` (`clocksync_shm`).
  Non-delegates attach and block up to `SYNC_TIMEOUT_SEC=60`
  (`Firefly.h:38`); on timeout they proceed **uncorrected with a warning**.
- Seqlock read fast path: `seq==0` ⇒ no sync ⇒ zero correction
  (`Utility.h:70-87`). Seqlock correctness + PI drift are unit-tested
  (`rpd_tracer/tests/test_seqlock.cpp`, `test_convergence.cpp` — build with
  `-DRPD_BUILD_TESTS=ON`).
- torchrun auto-configures rank/master (`docs/README_torchrun.md`);
  standalone uses `RPDT_CLOCKSYNC_RANK/_MASTER/_PORT` (env-only, bypasses
  rlog properties — FIXME `Firefly.h:29`).

## pid / gpu / device identity

- pid/tid = cached raw syscalls, `thread_local` (`Utility.h:26-36`).
- Roctracer normalizes deviceId by min driver-node-id
  (`RoctracerDataSource.cpp:56-82`).
- SMI deviceId vs `gpuId` renumbering under `ROCR_VISIBLE_DEVICES` is a known
  unsolved mapping issue (`AmdSmiDataSource.cpp:259-275`).
- Multi-node merge strides (pid_stride 10M / gpu_stride 1000) are applied at
  the node-0 receiver only; de-striding views `napi`/`nop` undo them
  (`utilitySchema.cmd:27-28`). Note `tools/multiplerpd2tracing.py` uses
  different ad-hoc offsets (`node_id*100000`) — don't mix the two conventions.
