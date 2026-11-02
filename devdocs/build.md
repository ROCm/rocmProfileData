# Build system

## Entry points

| Entry | What it really does |
|---|---|
| `cmake -B build -S . && cmake --build build -j && cmake --install build` | The real build (`CMakeLists.txt:4-6`) |
| `make` / `make install` / `make clean` | Thin wrappers around the cmake commands above (`Makefile:12-22`). They build nothing themselves. |
| `./install.sh` | apt deps (`sqlite3 libsqlite3-dev libfmt-dev nlohmann-json3-dev xxd`) + configure/build/install |

Build type forced to `Release` if unset (`CMakeLists.txt:10-12`); C++17,
`POSITION_INDEPENDENT_CODE ON` (`CMakeLists.txt:14-16`).

## CMake options

| Option | Default | Effect |
|---|---|---|
| `RPD_BUILD_TRACER` | ON | `rlog` + `rpd_tracer` subdirs (`CMakeLists.txt:59,67-79`) |
| `RPD_BUILD_REMOTE` | ON | `remote/` → `librpd_remote.so` (`:60,81-83`) |
| `RPD_BUILD_PYTHON` | ON | `pip install .` of `rocpd_python/` as an ALL target (`:61,89-100`) |
| `RPD_BUILD_VIEWER` | ON | `pip install .` of `rpd_viewer/` (depends on rocpd target, `:62,106-120`) — drags dash/plotly/openai into system python; turn OFF on shared machines |
| `RPD_ENABLE_CPPTRACE` | OFF | stack frames; needs `git submodule update --init cpptrace` (`rpd_tracer/CMakeLists.txt:123`) |
| `RPD_ENABLE_ROCM_TRACE_LITE` | OFF | compiles 3 RTL `.cpp` files into the tracer (`rpd_tracer/CMakeLists.txt:210-227`). **Only exists when `ROCM_ROOT_DIR` is non-empty** — silently absent on non-ROCm machines. |
| `RPD_BUILD_TESTS` | OFF | rpd_tracer clocksync unit tests (`rpd_tracer/CMakeLists.txt:292-296`) |
| `RPD_INSTALL_RPATH` | auto | bakes `${ROCM_ROOT_DIR}/lib` into installed `.so` RUNPATH (`CMakeLists.txt:42-49`) |

Note: the option is `RPD_BUILD_PYTHON`, not `BUILD_PYTHON`.

## Dependency discovery

`cmake/FindROCmRoot.cmake` (included `CMakeLists.txt:22`) resolves ROCm in
order: cache var `ROCM_ROOT_DIR` → `ROCM_PATH` → `ROCM_HOME` → python probe for
`_rocm_sdk_devel` (TheRock/rocm-sdk wheel) → `/opt/rocm`. It also defines
`rocm_find_package()`, which temporarily prepends the root + its
`lib/rocm_sysdeps` for **one** `find_package` call — rocm-sdk wheels vendor
private sqlite3/zstd/zlib/etc. that must never satisfy an unrelated
`find_package` (rationale `cmake/FindROCmRoot.cmake:94-104`). Use
`rocm_find_package()` only for SDK packages (hip, rocprofiler-sdk,
hsa-runtime64, nlohmann_json); plain `find_package()` otherwise.

| Dep | Where | Required? |
|---|---|---|
| SQLite3, fmt, Threads | `rpd_tracer/CMakeLists.txt:21-23` | yes |
| `xxd` | `rpd_tracer/CMakeLists.txt:88` | yes (schema codegen) |
| rocprofiler-sdk | `rpd_tracer/CMakeLists.txt:167-185` | optional; **skipped on ROCm 6.x** (perf) |
| hip / roctracer / CUPTI / CLR `hip_profiler_ext.h` | `:147-202,238-249` | optional; each enables its DataSource |
| cpptrace, rocm-trace-lite | submodules | opt-in |

No `ExternalProject`/`FetchContent` anywhere — everything is `add_subdirectory`
or a `pip install` custom target.

## How sub-projects are wired

| Component | Mechanism |
|---|---|
| `rlog` | git submodule **of this same repo, branch `rlog`** + `add_subdirectory`; FATAL_ERROR with `git submodule update --init rlog` hint if missing (`CMakeLists.txt:67-75`) |
| `cpptrace` | submodule, `add_subdirectory` EXCLUDE_FROM_ALL, `CPPTRACE_BUILD_SHARED=ON` (`rpd_tracer/CMakeLists.txt:125-136`) |
| `rocm-trace-lite` | submodule, **not** add_subdirectory — 3 source files pulled into `rpd_tracer_objs` (`rpd_tracer/CMakeLists.txt:211-227`) |
| `rocpd_python`, `rpd_viewer` | `add_custom_target(... pip install .)` + `install(CODE ...)` (`CMakeLists.txt:89-120`) |
| tools, raptor, examples, helpful_queries | not wired to anything |

## Generated files (never hand-edit)

`rpd_tracer/tableSchema.h` and `utilitySchema.h` are generated at build time
from `rocpd_python/rocpd/schema_data/{tableSchema,utilitySchema}.cmd` via
`xxd -i` (`rpd_tracer/CMakeLists.txt:92-118`; Make equivalent `rpd_tracer/Makefile:97-107`).
They are untracked build outputs; regenerate by rebuilding. Schema edits go in
the `.cmd` files — see [recipes/add-schema-table.md](recipes/add-schema-table.md).

## Targets worth knowing

- `rpd_tracer_objs` (OBJECT lib) → `librpd_tracer.so` (defines `RPD_TRACER_BUILD`,
  `-rdynamic` + `marker_shim.ver` version script) and `librpd_embedded.so`
  (same objects, no define) — `rpd_tracer/CMakeLists.txt:257-281`
- `rpdrun` — plain C delegate launcher (`:286-287`)
- `rlog_hub` → `librlog.so`, `rlog-config` CLI (`rlog/CMakeLists.txt:13-37`)
- `rpd_remote` → `librpd_remote.so` (`remote/CMakeLists.txt:4-11`)
- install runs `ldconfig` unconditionally (`rpd_tracer/CMakeLists.txt:322`,
  `remote/CMakeLists.txt:22`) — assumes root, staging-hostile

## Tests

- rlog: standalone `cmake -B build && ctest --test-dir build` — 4 tests
  (`rlog/CMakeLists.txt:74-103`; real-lib tests skip with rc 77). Discovered
  from the root build too (root `enable_testing()`, `CMakeLists.txt:53-57`).
- rpd_tracer: `ctest` only with `-DRPD_BUILD_TESTS=ON` (seqlock, measurement,
  regression, convergence — clocksync).
- RTL (submodule): `make test-cpu` (pytest, ~311 tests incl. source-audit
  greps), `make test-gpu` with a GPU.
- No test suite covers rpd_tracer itself beyond the clocksync units — verify
  tracer changes by recording a trace with `runTracer.sh` and querying it.

## CI reality check

`.github/workflows/ci.yml` triggers on push/PR to **`main`**, but the default
branch (origin and upstream) is **`master`** → CI effectively never runs.
When it does run it: inits only the `rlog` submodule, no ROCm (so **no GPU
backend is compiled**), and never runs `ctest`. Treat a green (missing) CI as
no signal; build locally.

## Legacy per-directory Makefiles — do not use

`rpd_tracer/Makefile` and `remote/Makefile` still exist but have genuinely
diverged from CMake (stale cpptrace path, RTL auto-enable, `-lrlog` from
`/usr/local`, c++11/c++17 mismatch). They will produce a **different** library.
CMake is the only maintained path; keep the Makefiles in sync only if you
touches something they share deliberately.
