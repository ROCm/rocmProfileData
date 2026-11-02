# rocpd python package & tools/ scripts

## rocpd (`rocpd_python/`, pip name `rocpd`, stdlib-only, raw sqlite3 — no SQLAlchemy)

The schema owner and the post-processing toolkit. Install: `cd rocpd_python &&
make install` (= `pip install .`); also pip-installed by root CMake.

| Module | Purpose / entry point |
|---|---|
| `rocpd/schema.py` | `RocpdSchema` applies the `.cmd` DDL scripts; `python -m rocpd.schema --create trace.rpd` makes an empty trace (`:45-57`) |
| `rocpd/schema_data/*.cmd` | **authoritative DDL** — tableSchema, utilitySchema, indexSchema, index2Schema. C++ tracer embeds table+utility via `xxd -i` at build time |
| `rocpd/importer.py` | `RocpdImportData` — id counters + string-interning for building rpds: `initNew`/`resumeExisting`/`getStringId`/`commitStrings` (`:30-86`) |
| `rocpd/metadata.py` | `Metadata.get/set/clear/listAll` over `rocpd_metadata` |
| `rocpd/tracing.py` | `.rpd` → Chrome/Perfetto JSON (`generate_rpd_json`) and Kineto JSON (`generate_kineto_json`); queue-depth delta-scan; **holds the only schema-version compat shim** (see [concepts/rpd-schema.md](../concepts/rpd-schema.md)); runnable as module |
| `rocpd/subclass.py` | `createSubclassTable()` — promote `k=v|...` args into real columns: `rocpd_<name>{api,op}` + view (`:23-120`) |
| `rocpd/call_stacks.py` | `ext_callstack` parent/child expansion; two impls (python replay vs SQL window-fn+CTE, `:5-11`); `python -m rocpd.call_stacks trace.rpd [--sql]` |
| `rocpd/autograd.py` | `ext_autogradapi` + `autogradKernel` views (torch op → kernel attribution); `python -m rocpd.autograd trace.rpd` |
| `rocpd/graph.py` | HIP/CUDA graph capture → `ext_graph`, `ext_graph_kernelapis`, `graphLaunch`/`graphKernel` views; `python -m rocpd.graph trace.rpd` |
| `rocpd/deserialize.py` | promote roctx `api=... arg=...` marker payloads into real api/op rows |
| `rocpd/rocprofiler_import.py` | legacy rocprof txt → rpd importer (successor of deleted `shelve/rocprof2rpd.py`), same argparse surface (`--ops_input_file` etc.) |
| `rocpd/strings.py`, `rocpd/util/chrometracing.py` | string cleanup; Django-era JSON writer (legacy) |

Pattern for readers in this repo: open `sqlite3`, join id columns to
`rocpd_string`/`rocpd_ustring`. There is **no** upstream-style
`read_*_schema`/`*_iterator` API here. Post-processors mutate the trace file
(RW connection, guarded by `*_::Table` tags in `rocpd_metadata` so they're
idempotent).

## tools/ — standalone scripts (not installed, not in build)

Run from a checkout with `rocpd` importable. Note: the upstream
ROCm/rocmProfileData tool names (rpd2json, rpd2tb, rocpdump, ...) **do not
exist in this repo** — this is the real list:

| Tool | One-liner |
|---|---|
| `rpd2tracing.py` | one rpd → Chrome/Perfetto JSON; thin shim on `rocpd.tracing` (`:13,29`); `--start/--end` in µs or `%` |
| `multiplerpd2tracing.py` | merge N rpds (per-node) into one Chrome trace; ad-hoc pid offset `node_id*100000` (`:73`) — differs from DB-level `pid_stride`; prefer `napi`/`nop` views for DB work |
| `rpd_trim.py` | destructively delete events outside a time window (`--dryrun` supported), then `cleanStrings` |
| `rpd2table.py` | CSV summary of APIs/kernels scoped to named `UserMarker` ranges (temp `ext_marker`; v3-only — queries `rocpd_ustring` directly, `:40`) |
| `rpd_autograd_summary.py` | HTML "kernels by autograd operator" report; requires `python -m rocpd.autograd` first |
| `compare_rpds.py` | RCCL-collective comparison across a directory of rpds (clocksync validation aid) |
| `rpt2rpd.py`, `nvvp2rpd.py` | niche importers (legacy HIP_TRACE text; NVIDIA nvvp export — self-described "may stay early stage") |

`raptor/raptor.py` (standalone, not in tools/): ROI slicing, kernel
categorization (`raptor_cat_vllm.json`), idle-gap and variability analysis;
`raptor/README.md` + own pytest suite.

## Adding things

- New analysis script: add to `tools/`, argparse, import `rocpd` rather than
  duplicating schema knowledge.
- New derived data that belongs to the format: add a rocpd post-processor
  module writing an `ext_*` table guarded by a metadata tag — follow
  `call_stacks.py` as the template, and see
  [recipes/add-schema-table.md](../recipes/add-schema-table.md).

## Example traces

`examples/rocm-profile-data/` is the live blog tutorial (torch matmul;
checked-in `results/*.rpd`). Its embedded schema dump is **v2-era — do not
cite it** as current schema.
