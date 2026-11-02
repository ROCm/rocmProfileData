# Architecture

rocmProfileData = **RPD**: an SQLite-based trace file format plus the
collection and analysis ecosystem around it. One sentence per layer:

```
 application (HIP / ROCm / CUDA / plain CPU)
     │ injected:  LD_PRELOAD=librpd_tracer.so   (constructor, rpd_tracer/rpd_capi.cpp:90)
     │            or link librpd_embedded.so    (manual rpdstart/rpdstop/rpdflush)
     │            or rpdTracerControl (python)  (CDLL + RPDT_AUTOSTART=0)
     │            or LD_PRELOAD=librpd_remote.so + `rpdRemote start <pid>` (SIGUSR1)
     ▼
 Logger singleton (rpd_tracer/Logger.cpp:128) — discovers DataSources via
     dlsym("<Name>DataSourceFactory"), exactly one GPU provider wins
     ▼
 DataSources (Clr | Rocprof | Roctracer | Rtl | Cupti + markers/telemetry)
     ▼ write rows through BufferedTable ring buffers + per-table writer threads
 trace.rpd — SQLite (WAL), rocpd_* schema, schema_version=3
     ▼
 consumers: sqlite3 CLI views (api/op/top/busy) · rpd-viewer (Dash web app,
 incl. read-only SQL console + LLM chat) · tools/rpd2tracing.py (Chrome/
 Perfetto JSON) · rocpd python package · raptor
```

## Component map

| Component | Path | Lang | What it is | Build |
|---|---|---|---|---|
| **rpd_tracer** | `rpd_tracer/` | C++17 | The profiler. Builds `librpd_tracer.so` (LD_PRELOAD) and `librpd_embedded.so` (manual control) from identical objects; `rpdrun` delegate launcher; `rpdTracerControl.py`. See [subsystems/rpd-tracer.md](subsystems/rpd-tracer.md). | core |
| **rlog** | `rlog/` (submodule = same repo, `rlog` branch) | C++14 | Annotation-routing hub (`librlog.so`): app↔tool decoupling, zero-cost when no tool listens; plus `$HOME/.rlog.db` property store + `rlog-config` CLI. Mandatory tracer dependency. [subsystems/rlog.md](subsystems/rlog.md). | core |
| **rocpd** | `rocpd_python/` | Python | Canonical schema (`schema_data/*.cmd`), importers, post-processors (call stacks, autograd, graphs), Chrome/Kineto JSON export. [subsystems/python-and-tools.md](subsystems/python-and-tools.md). | core (pip) |
| **rpd_viewer** | `rpd_viewer/` | Python/Dash | Web viewer, ~21 auto-discovered pages, live-trace mode, SQL console, LLM chat over the `.rpd`. [subsystems/viewer.md](subsystems/viewer.md). | core (pip) |
| **remote/** | `remote/` | C++11 | Tiny SIGUSR1/SIGUSR2 → `rpdstart/rpdstop/rpdflush` trigger (`librpd_remote.so` + `rpdRemote` CLI). Unrelated to `RemoteDataSource` — see naming trap in [gotchas.md](gotchas.md). | core |
| **rocm-trace-lite** | `rocm-trace-lite/` (submodule, sunway513) | C++ | Standalone HSA-tools kernel profiler (zero dep on roctracer/rocprofiler-sdk); RPD-schema `.db` output; `rtl` CLI. Also embedded into the tracer as `RtlDataSource` (opt-in). [subsystems/rocm-trace-lite.md](subsystems/rocm-trace-lite.md). | opt-in |
| **cpptrace** | `cpptrace/` (submodule) | C++ | Vendored stack-frame capture for `rocpd_stackframe` (opt-in, `RPD_ENABLE_CPPTRACE=ON` + `RPDT_STACKFRAMES=1`). | opt-in |
| **raptor** | `raptor/` | Python | Standalone post-processing CLI + pandas parser (ROI slicing, kernel categorization, variability). Semi-maintained, not in build; own tests. | none |
| **tools/** | `tools/` | Python | Standalone scripts (rpd2tracing, trim, table, importers). Not installed; run from a checkout with `rocpd` importable. | none |

### Legacy / dead directories (do not extend; do not cite as current)

| Path | Status |
|---|---|
| `rpd_dash/` | **Deleted in git** (`0d2144b`, folded into `rpd_viewer/`); only stale untracked `build/` leftovers on disk. A *Django* predecessor at repo root was deleted in `3819abb`. |
| `shelve/` | Dead — `rocprof2rpd.py` superseded by `python3 -m rocpd.rocprofiler_import`. |
| `rtg_tracer/` | Dead submodule pointer (uninitialized, zero references). |
| `autocoplite/` | Stale third-party visualization submodule; README instructions conflict with the CMake build. |
| `autograd/` | 2021-era PyTorch patch, legacy. Modern path: `rocpd.autograd` builds `ext_autogradapi` from an existing rpd. |
| `hipMarker/` | Legacy marker extension (actually emits roctx now); superseded by rlog / rpdTracerControl ranges. |
| `helpful_queries/` | Useful example SQL, not code; partially superseded by `docs/perf_playbooks/`. |
| `examples/rocprofiler/` | Dead 2018-era rocprof v1 patch. |

## Key structural facts

1. **The schema is Python-owned, C++-consumed.** `rocpd_python/rocpd/schema_data/
   {tableSchema,utilitySchema}.cmd` are the single source of truth; CMake
   bakes them into C++ byte arrays with `xxd -i` (`rpd_tracer/CMakeLists.txt:92-118`,
   consumed by `rpd_tracer/Schema.cpp:37-58`). There is **no `rpd_schema.h`** —
   see [concepts/rpd-schema.md](concepts/rpd-schema.md).
2. **No build-time backend selection.** Every detected GPU DataSource is
   compiled in; `Logger::init()` picks exactly one at runtime via `dlsym` of
   exported `<Name>DataSourceFactory` symbols, first match in a priority list
   wins (`rpd_tracer/Logger.cpp:155-231`). Rationale comment:
   `rpd_tracer/CMakeLists.txt:4-19`.
3. **Injection is LD_PRELOAD + ELF constructor**, not rocprofiler-register.
   The tracer wins the shutdown race by interposing `atexit/__cxa_atexit`
   (`rpd_tracer/rpd_capi.cpp:124-146`). Details:
   [concepts/injection-and-interposition.md](concepts/injection-and-interposition.md).
4. **Markers are provided by the tracer itself.** `librpd_tracer.so` exports
   `roctxMarkA/roctxRangePushA/roctxRangePop` and the NVTX equivalents via a
   linker version script (`rpd_tracer/marker_shim.ver`) so apps can link
   markers without libroctx64/libcupti present.
5. **Everything is one SQLite file.** Multi-process and multi-node traces
   append to a single `.rpd` using id-space striding (sessionId<<32, nodeId<<48)
   and a TCP aggregation delegate — [subsystems/remote-and-distributed.md](subsystems/remote-and-distributed.md).
6. **Config precedence** for every tracer knob: `rpd_setConfig()` > env `RPDT_*`
   > rlog property (`$HOME/.rlog.db`, domain `rpd_tracer`) > default
   (`rpd_tracer/Config.cpp:23-33`). Some knobs are env-only by omission.
7. Distributed stack (rpdrun, ChronoSync/Firefly, NetWriterBackend,
   RemoteDataSource) landed 2026-09 in one 12-PR "distdev" series — recently
   landed, expect drift.

## Where user-facing docs live (link, don't duplicate)

`docs/README_tracer_quickstart.md` (canonical quickstart — **not** the root
README, see [gotchas.md](gotchas.md)), `docs/README_embedded.md` (C API +
embedded mode), `docs/README_distributed.md` + `docs/README_clocksync.md` +
`docs/README_torchrun.md` (multi-node), `docs/README_gpucounters.md`,
`docs/perf_playbooks/*` (analysis recipes),
`rpd_viewer/RPD_INFO.md` (schema reference), `rlog/README.md`,
`rocm-trace-lite/docs/`.
