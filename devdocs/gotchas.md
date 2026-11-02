# Gotchas, dead code, stale docs

Quick "don't get shot" list, consolidated from research. Verify still true
before acting — several are fixable bugs, and this doc ages.

## Live bugs / traps in master (@ 192625e)

1. **SMI datasource silently never loads.** Default factory list asks for
   `"RocmSmiDataSourceFactory"` (`Logger.cpp:164`) but the exported symbol is
   `AmdSmiDataSourceFactory` (`AmdSmiDataSource.cpp:316`); dlsym miss is a
   silent skip ⇒ `rocpd_monitor` empty, GPU-Monitor viewer page dead.
   Workaround: `RPDT_DATASOURCES_PRIORITY=AmdSmiDataSource` (priority adds
   unknown names). The file also isn't in `RPD_COMMON_SRCS` — double-dead;
   docs still describe SMI sampling as live (`docs/perf_playbooks/freq-capture.md` etc.).
2. **Backend-priority docs contradict code**: CLR wins by default whenever
   `hip_profiler_ext.h` was present at build (`Logger.cpp:155-165`);
   `docs/README_tracer_quickstart.md:79` says Rocprof. Check the
   `process_datasource` metadata row, not the docs.
3. **CI is off**: workflow triggers on `main`; branches are `master`
   (origin+upstream). When it runs it compiles no GPU backend and never runs
   ctest. `CONTRIBUTING.md` also says "PR against main". Green ≠ built.
4. **`ensureSchema` never migrates** (`Schema.cpp:34-42`): pre-existing file of
   any version is accepted as-is; writing v3 rows into a v2 file yields a
   mixed file. Most readers detect by feature probes, but
   `tools/rpd2table.py:40` breaks on ≤v2, and v1 files have **no**
   `schema_version` row (`None` ⇒ treat as ≤2, `rocpd/tracing.py:43`).
5. **Readers aren't uniformly read-only.** rpd_viewer's main path opens RW and
   sets WAL (`util/db.py:17-24`) — it can create side files on a clean db.
   Only chat/query/`/tracedata` use `mode=ro`. Post-processors mutate by
   design. Keep `*-shm/-wal` with the `.rpd`.
6. **RTL `rtl convert --format rocprofv3` is broken**: selects
   `correlation_id` columns the C++ schema never creates
   (`cmd_convert_rocprofv3.py:69,179` vs `trace_db.cpp:80-100`); its tests
   ALTER the schema to hide it.
7. **RTL multi-process merge drops api/kernel/copy rows + `roctxId`**
   (`cmd_trace.py:274-324`).
8. **Remote trigger stop-path leaks its refcount if the start dlopen failed**
   (`remote/Remote.cpp:51-69`); dlopen inside a signal handler is unsafe —
   known, don't build on it.
9. Two node-id→id encodings coexist: metadata strides (pid 10M/gpu 1000,
   views `napi`/`nop`) vs `multiplerpd2tracing.py`'s `node_id*100000`. Don't
   mix.

## Dead / `#if 0` code in rpd_tracer (don't "fix" callers)

`RocprofDataSource::counter_buffer_callback` (`:991-1095`, never registered);
`BarrierOpTable` (`Table.h:235-256`); `hip_activity_callback` + hipGraphLaunch
kernel rows in Roctracer (`:740-796`, `:336-360`); `OpTable::associateDescription`
no-op; `ApiIdList::loadUserPrefs` stubs; CUPTI init commented enables;
`RtlDataSource::current_roctx_id` stub + no-op start/stop/flush;
`RocmSmiDataSource` itself (see bug #1). In RTL: `record_copy` dead
(async-copy capture unimplemented).

## Stale docs — trust these files less

- `README.md`: cpptrace "built by make" (false — opt-in option); autocoplite
  `make install` venv instructions collide with top-level `make install`;
  viewer never mentioned; dep lists drift vs `install.sh` and CI
  (CI adds `libzstd-dev`). Real quickstart: `docs/README_tracer_quickstart.md`.
- `rpd_tracer/README.md`: dlopen example uses removed mangled symbols
  (`_Z12init_tracingv`); actual C API is `rpd_tracer.h:9-16`.
- `tools/README_step2.txt` names `rocprof2rpd.py` — deleted; now
  `python3 -m rocpd.rocprofiler_import`.
- `examples/rocm-profile-data/` schema dump is v2-era.
- `rlog/README.md` says `libroctx64.so`; code default is
  `librocprofiler-sdk-roctx.so`. Its OPTIMIZATIONS.md reference is dangling.
- RTL `PLANNING.md`/`REVIEW.md` are historical (thread-per-dispatch never
  shipped; ADR-001 & perfetto docs partially superseded).
- `rocm-trace-lite/REVIEW.md` items mostly fixed — check before repeating them.

## Directory traps

- `rpd_dash/`, `rpd_viewer/build/`, `rocpd_python/build/`, `*/egg-info/`,
  `rpd_tracer/*.o,*.d,rpdrun,tableSchema.h,utilitySchema.h`, `remote/*.o,*.so`,
  root `trace.rpd*`/`.rlog.db*` — untracked build litter, not source.
  `rpd_dash` is deleted in git entirely.
- `rlog/` edits go to branch `rlog` of this repo (PR base!), then bump the
  submodule pointer on master.
- `rtg_tracer/`, `autocoplite/` submodules point at third-party personal repos,
  uninitialized, unreferenced — treat as remove-on-contact candidates, not
  features.
- Generated schema headers: edit `rocpd_python/rocpd/schema_data/*.cmd`, never
  the `xxd` output.
- `install(CODE ldconfig)` fires on every `cmake --install` — needs root even
  for staging.

## Units & data-semantics gotchas (analysis side)

ns in DB / µs in tool args & `top` view; `busy.Busy` can exceed 1.0 with
overlapping queues; negative GPU durations possible; `rocpd_api.args_id`
joins `rocpd_ustring`, every other `*_id` joins `rocpd_string` (top
schema-error source — RPD_INFO.md §2); monitor table is RLE (spans, not
samples).
