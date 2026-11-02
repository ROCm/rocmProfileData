# rlog — annotation routing hub

Standalone library on branch **`rlog` of this same repo**, consumed as a
self-referencing submodule (`.gitmodules:10-13`). PRs must target the `rlog`
branch, not master (`rlog/CONTRIBUTING.md:24-26`). Has its own CI (runs
build+ctest on that branch).

Purpose: pass annotation messages (marks, ranges, properties) between
applications and profiling tools with **zero cost when no tool is listening**
(`rlog/README.md:7-8`). It is the modern replacement for calling roctx/NVTX
directly. It is NOT a logger that writes files — routing only; tools that
record are separate.

## Shapes

```
 app (compiles include/rlog/client.h + src/rlog.cpp; no link-time dep)
   │  src/rlog.cpp dlopens backends at runtime:
   │    ├─ librlog.so  (the hub)            ← rlog native
   │    ├─ roctx lib   (librocprofiler-sdk-roctx.so by default)
   │    └─ NVTX lib    (libcupti.so)
   ▼
 librlog.so  — rlog::Hub singleton, C ABI exports (src/Hub.cpp:14-44)
   ▼ registered rlog::Logger objects (fan-out; lock-free fast path when
     exactly one logger is registered, Hub.cpp:54-55,84-112)
 rpd_tracer::RlogDataSource (registers itself on startTracing,
   rpd_tracer/RlogDataSource.cpp:106-109) → rocpd_api rows
```

## Public API

- Client `namespace rlog` (`include/rlog/client.h`): `init()` (not
  thread-safe), `mark(domain,category,apiName,args)` + shorter overloads,
  `rangePush/rangePop`, `isActive()`, `registerActiveCallback`,
  `setDefaultDomain/Category` (not thread-safe), `getProperty`, per-backend
  `setEnabled(Rlog|Roctx|Nvtx)`.
- `rlog::Range` / `rlog::Scope` RAII + `rlog::fmt` (`include/rlog/Range.h`).
  **Critical performance contract:** gate is a caller-cached
  `std::atomic<bool>`; expensive args must be passed as *callables* —
  eager string args cost ~509ns/range even while logging is OFF, deferred
  cost 0; guarded emit ~5ns (`Range.h:8-23`, `rlog/benchmarks/README.md:127-164`).
  The gate is deliberately sampled twice per range, never latched, so a
  mid-range tool attach still closes orphan ranges.
- Tool side (`include/rlog/Logger.h`, `Hub.h`): implement `mark/rangePush/rangePop`,
  `Hub::singleton().addLogger()` (ref-counted).
- Python: `rlog/python/rlog.py` ctypes client (`RlogClient`, `range()` context
  manager, `range_decorator`), installed to site-packages by CMake
  (`rlog/CMakeLists.txt:45-52`).

## Property database (the only SQLite in rlog)

`$HOME/.rlog.db`, table `properties(domain, property, value)`
(`src/PropertyDb.cpp:64-70`); WAL + `BEGIN IMMEDIATE`; per-process cache means
**a property read once keeps its value for the process lifetime**
(`PropertyDb.cpp:98-116`). CLI: `rlog-config get|set|list domain:property`
(`src/rlog-config.cpp`). This is the third tier of the tracer's config system
(`rpd_tracer/Config.cpp:32`) and how `runTracer.sh` reads defaults
(`rpd_tracer/runTracer.sh:23-26`).

## Env vars (also stored as properties in domain `rlog`)

| Var | Effect | Ref |
|---|---|---|
| `RLOG_ROCTX_LIBPATH` | which lib provides roctx symbols | `rlog/src/rlog.cpp:112` |
| `RLOG_NVTX_LIBPATH` | which lib provides NVTX symbols | `rlog/src/rlog.cpp:113` |
| `RLOG_FORCE_ROCTX` / `RLOG_FORCE_NVTX` | force backend on even with no listener (makes `isActive()` true) | `rlog/src/rlog.cpp:114-134` |

## Build / test / bench

- Standalone: `cmake -B build && cmake --build build && ctest --test-dir build`.
  Deps: CMake≥3.14, C++14, SQLite3 only. Tests skip (rc 77) when real
  roctx/nvtx libs absent (`rlog/CMakeLists.txt:88-103`).
- Benchmarks (not in ctest): `make bench_loopback bench_external bench_roctx
  bench_guard bench_range` — interpret with `rlog/benchmarks/README.md`.
- Via root build: `add_subdirectory` when `RPD_BUILD_TRACER=ON`; the tracer
  both compiles `src/rlog.cpp` in as its client shim and links the `rlog_hub`
  target for `RlogDataSource` (`rpd_tracer/CMakeLists.txt:79-83`).

## Watch out

- The README's forwarding table still says roctx lib is `libroctx64.so`; code
  default is `librocprofiler-sdk-roctx.so` (`rlog/src/rlog.cpp:112`).
- `getVersion()/localVersion()` are commented-out FIXME stubs
  (`include/rlog/client.h:30-31`).
- Historical confusion: an old "RLogger" prototype (rlog as a .rpd writer)
  exists only on unmerged branch `origin/rlog-maybe-important`; the "BufferPool"
  from PR #127's title no longer exists on master. Current rlog = routing hub.
- Clocksync deliberately bypasses rlog properties (plain getenv; FIXME
  `rpd_tracer/Firefly.h:29`).
