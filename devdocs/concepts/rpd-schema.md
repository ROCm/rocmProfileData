# The RPD schema

Format = SQLite file (conventionally `trace.rpd`) with `rocpd_*` tables,
WAL journal, int64-ns timestamps. Current `schema_version = 3`.

**Canonical analysis-facing reference: `rpd_viewer/RPD_INFO.md`** (units,
string-join rules, api→op bridge, views, sentinels, args-as-JSON, queue-depth
recipes; section TOC at its head; also the viewer chat system prompt). This
file covers what RPD_INFO.md deliberately omits: where the schema lives,
versioning, and how it evolves.

## Where it lives (single source of truth chain)

```
rocpd_python/rocpd/schema_data/tableSchema.cmd    ← EDIT HERE (tables + version rows)
rocpd_python/rocpd/schema_data/utilitySchema.cmd  ← EDIT HERE (views)
        │ python: RocpdSchema applies .cmd directly (rocpd/schema.py:30-43)
        │ C++:  cmake runs `xxd -i` → build-dir tableSchema.h/utilitySchema.h
        │       (rpd_tracer/CMakeLists.txt:92-118), applied idempotently by
        │       Schema.cpp:17-62 under BEGIN EXCLUSIVE
        └ indexSchema.cmd / index2Schema.cmd: python-side optional indexes
```

There is **no `rpd_schema.h`** header and no rlog schema — the tracer's schema
headers are generated build outputs (untracked; hand edits die on `clean`).
`rlog`'s only SQLite is its `$HOME/.rlog.db` property store, unrelated.

## Version history & detection

| Ver | Marker in `rocpd_metadata` | Change |
|---|---|---|
| v1 | **no row at all** | op-keyed subclasses `rocpd_kernelop/copyop`; signal-graph tables (`rocpd_op_inputSignals`, `rocpd_barrierop`, `rocpd_kernelcodeobject`, `op.completionSignal`) |
| v2 (`e749a93`) | `("schema_version","2")` | kernel/copy re-keyed to the **launching API**: `rocpd_kernelapi`/`rocpd_copyapi`; views kernel/copy/copyop |
| v3 (`9ac0786`/`a8b269d`) | `("schema_version","3")` (`tableSchema.cmd:20`) | `rocpd_api` + `category_id`,`domain_id`; `rocpd_ustring` split; adds `rocpd_monitor`, `rocpd_counter`, `rocpd_stackframe`; drops v1/v2 tables; seeds `gpu_stride`=1000, `pid_stride`=10000000 |

Detection is **feature-based in nearly every reader** (v1 has no version row):

- `rocpd/tracing.py:34-47` — the only `schema_version` reader: if absent or
  ≤2, creates `CREATE TEMPORARY VIEW rocpd_ustring AS SELECT * FROM rocpd_string`
  so v3 queries run unchanged; `_has_v3_schema()` probes `domain_id` column
  (`:457-462`).
- Viewer probes columns/tables (`rpd_viewer/rpd_viewer/util/db.py:110-150`).
- Writer side: `Schema.cpp:34-42` only checks whether `rocpd_string` exists —
  **no migration ever happens**; appending v3 data into a v2 file produces a
  mixed file no reader fully handles (see [gotchas.md](../gotchas.md)).

## Table set (v3)

Core: `rocpd_metadata` (tags incl. session/`process_datasource`/delegate
markers/generator guards), `rocpd_string` (**deduped** interned strings —
every `*_id` points here), `rocpd_ustring` (**never deduped** — args payloads;
`rocpd_api.args_id → rocpd_ustring` is the single most common join mistake),
`rocpd_api` (CPU events: pid,tid,start,end,name/category/domain/args ids),
`rocpd_op` (GPU events: gpuId,queueId,sequenceId,start,end,description,opType;
`rocpd_op.api_id` → launching api row), `rocpd_api_ops` (correlation bridge),
`rocpd_kernelapi` (grid/wg/lds/kernelName per launch), `rocpd_copyapi`,
`rocpd_monitor` (RLE device telemetry), `rocpd_counter` (per-dispatch HW
counters), `rocpd_stackframe` (cpptrace frames).

Views (`utilitySchema.cmd`): `api`, `op`, `top`, `busy`, `kernel`, `copy`,
`copyop`, `stackframe`, `counter`, `counter_summary`, and multi-node
de-striding `napi`/`nop`.

Post-processing `ext_*` tables (created only by rocpd modules, guarded by
metadata tags): `ext_callstack` (+ inclusive/exclusive views),
`ext_autogradapi` (+ `autogradKernel`), `ext_graph`/`ext_graph_kernelapis`;
generic subclass tables `rocpd_<name>{api,op}` via `rocpd.subclass`.

## Time & id conventions

- `start`/`end` = int64 **nanoseconds**, monotonic domain; only deltas are
  meaningful; GPU durations can be negative (hardware clock quirks —
  RPD_INFO.md §1).
- Row `id` encodes session: per-process offset `sessionId << 32`
  (`rpd_tracer/Storage.cpp:36-45`); annotation ids occupy bit 31+
  (`Storage.h:61`); node id adds bits 48–62 plus pid/gpu strides — full
  layout in `rpd_tracer/RemoteDataSource.cpp:24-49`.

## Changing the schema

Follow [recipes/add-schema-table.md](../recipes/add-schema-table.md).
Short version: edit the `.cmd` files, never the generated headers; keep
additive-only (old files are never migrated); update RPD_INFO.md (chat
prompt!) and the relevant consumer probes.
