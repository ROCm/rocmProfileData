# rocm-trace-lite (RTL)

Git submodule (`sunway513/rocm-trace-lite`, pinned v0.3.6-12; package version
0.3.7). Written from scratch; **never shares code with rpd_tracer in the other
direction and never writes `.rpd`** — it writes its own SQLite `.db` using the
RPD `rocpd_*` schema (`src/trace_db.cpp:78-159`).

Its own docs are good: `rocm-trace-lite/docs/{quickstart,how_it_works,
cli_reference,output_format,ADR-001-hipgraph-safety,multi_gpu}.md`. This file
adds: the map, what the docs get wrong, and the embed contract.

## What it is

A kernel profiler whose only runtime deps are `libhsa-runtime64` + sqlite3 —
zero roctracer/rocprofiler-sdk/libamdhip64/libroctx64, enforced by CI + an
`ldd` check (`Makefile:34`). Primary mechanism is **not** LD_PRELOAD: it is an
**HSA tools library** loaded via `HSA_TOOLS_LIB`; exported `OnLoad()` swaps
`hsa_queue_create`/`hsa_executable_freeze` in the HSA API table
(`src/hsa_intercept.cpp:742-763`), then times kernels by injecting HSA signals
into AQL dispatch packets, resolved by one completion-worker thread.
LD_PRELOAD + `dlsym(RTLD_NEXT)` is the secondary mechanism for `RTL_MODE=hip`
HIP-API capture (`src/hip_api_intercept.cpp`).

Modes (`RTL_MODE`, parsed `src/hsa_intercept.cpp:769-788`):
`lite` (default; skips packets that already carry a signal) | `standard`
(`default` = alias) | `full` | `hip` (= standard **plus** HIP-API; docs saying
hip mode has "no HSA queue interception" are wrong).

## Source map

| File | Role |
|---|---|
| `src/hsa_intercept.cpp` (842L) | OnLoad/OnUnload, table swap, queue intercept cb, signal injection, single completion worker (`:197-304`), lock-free Vyukov MPMC signal pool (cap 4096, `:80-158`), shutdown (`:658-730`) |
| `src/hip_api_intercept.cpp` (648L) | 21 HIP wrappers, re-entrancy guard, RTLD_NEXT + RTLD_NOLOAD dlopen fallback (fork/exec-safe resolution, `:147-187`) |
| `src/trace_db.cpp/.h` | SQLite writer (WAL, `synchronous=OFF`, 1000-row transactions), schema+views, embedding callback hooks, `rtl_trigger_shutdown()` |
| `src/roctx_shim.cpp` | built-in roctx symbols, thread-local stacks |
| `rocm_trace_lite/` | `rtl` CLI: `cmd_trace.py` orchestrates (sets `HSA_TOOLS_LIB`+`RTL_OUTPUT=%p`+`LD_PRELOAD`, per-PID files, merge), `cmd_convert*.py` JSON export |

## Embedding contract (RPD side)

`rpd_tracer/CMakeLists.txt:210-227` compiles RTL's 3 core `.cpp` files into
`librpd_tracer.so` with `-DRPD_ENABLE_ROCM_TRACE_LITE=ON` (OFF by default).
`rpd_tracer/RtlDataSource.cpp` registers kernel/API event callbacks, maps them
into RPD's tables, auto-sets `HSA_TOOLS_LIB` via `dladdr` (`:163-168`), forces
the legacy HSA load path (`HSA_TOOLS_ROCPROFILER_V1_TOOLS=1`, `:170-176` —
needed because the tracer also exports `rocprofiler_configure`, which would
otherwise satisfy rocprofiler-register), and **must** call
`rtl_trigger_shutdown()` before finalizing its own storage
(`RtlDataSource.cpp:196-198`, `src/trace_db.h:132-136`). In callback mode RTL
opens no SQLite file (`src/hsa_intercept.cpp:750-753`).

## Facts not in RTL's docs (verified in source)

- **API→kernel correlation is NOT wired.** `CorrelationMap` exists but
  `pop()` is never called; correlation_ids are generated everywhere but never
  persisted (`src/hip_api_intercept.cpp:26-33` "Phase 2 — NOT YET WIRED";
  `src/trace_db.cpp:309`). Only roctx ranges persist (as `roctxId`).
- **Bug:** `rtl convert --format rocprofv3` SELECTs `correlation_id` columns
  that the C++ schema doesn't have (`rocm_trace_lite/cmd_convert_rocprofv3.py:69,179`
  vs `src/trace_db.cpp:80-100`); unit tests mask it with `ALTER TABLE`.
- **Multi-process merge drops `rocpd_api`/kernelapi/copyapi rows and `roctxId`**
  — merge only copies string/op/metadata (`rocm_trace_lite/cmd_trace.py:274-324`).
- `queueId` in `rocpd_op` is the `hsa_queue_t*` host pointer; the real HWQ
  address lives in the dispatch-info string used for Perfetto HWQ tracks
  (`src/hsa_intercept.cpp:537`, `cmd_convert.py:62-97`).
- `record_copy` has no caller — async copy capture is unimplemented
  (`src/trace_db.cpp:340`).
- Shutdown: registered via `std::atexit` inside `OnLoad`, idempotent via
  atomic exchange; drain forwards original signals to avoid app hangs
  (`src/hsa_intercept.cpp:658-730`).
- Doc drift: ADR-001's "default = full profiling" predates lite-by-default;
  `rocm-trace-lite/docs/perfetto.md` predates HWQ-address tracks;
  `cli_reference.md` lacks `--mode`; stale `rpd_lite.*` header comments.
- `PLANNING.md` and `REVIEW.md` are historical — most REVIEW P0s are fixed and
  test-enforced (e.g. no `.detach()`, `tests/test_completion_worker.py:14-22`);
  don't trust PLANNING's structure claims.

## Build / test

- C++: `make` → single `librtl.so`; needs ROCm headers + libsqlite3-dev;
  `make all` runs the forbidden-deps `ldd` guard. `make install` runs
  `ldconfig` (packaging-unsafe, known).
- Python: `pyproject.toml`/`setup.py` ship the `.so` (or JIT-build at install,
  `rocm_trace_lite/_build.py`); `build_wheel.sh` for wheels.
- Tests: `make test-cpu` (pytest, incl. source-audit greps); `make test-gpu`
  builds a hipcc workload. `profiler_perf_bench/` = cross-profiler overhead
  suite (`profiler-bench` CLI); `repro/` = minimal reproducer of the ROCm
  intercept-queue SEGFAULT behind ADR-001 (fixed upstream ROCm 7.13+).

## Env

`HSA_TOOLS_LIB` (set by `rtl trace`), `LD_PRELOAD` (always set by CLI for
roctx symbol visibility), `RTL_OUTPUT` (`%p` supported; legacy alias
`RPD_LITE_OUTPUT`), `RTL_MODE`, `RTL_DEBUG=1|2`, (undocumented:
`RTL_LIB_PATH` for the perf-bench adapter). No other getenv exists.

No multi-node support anywhere ("rtl_distrib" is an unfetched parent-repo
branch touching viewer/python, not RTL). Scale-out = one merged `.db` per node
yourself.
