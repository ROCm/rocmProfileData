# Recipe: add (or change) a schema table / view

Rule zero: the format is append-only in practice — **no migration code
exists** and old files are never rewritten (`rpd_tracer/Schema.cpp:34-42`).
Changes must be additive: a v3 reader still works on older files (feature
probe first), and an old reader ignores what it doesn't know.

## A. Decide which layer owns the new data

| Data kind | Where it goes |
|---|---|
| Core trace content something a DataSource emits during collection | real `rocpd_*` table (steps B–D) |
| Derived/computed from existing rows, after the fact | `ext_*` table written by a rocpd post-processor, guarded by a `rocpd_metadata` tag — template: `rocpd_python/rocpd/call_stacks.py:225-294` |
| Just a nicer query | view in `utilitySchema.cmd` |
| Per-run key/value | existing `rocpd_metadata` via `rocpd/metadata.py` — often no schema change needed |

## B. Edit the DDL (single source of truth)

`rocpd_python/rocpd/schema_data/tableSchema.cmd` (tables) and/or
`utilitySchema.cmd` (views). Notes:
- All DDL is `CREATE ... IF NOT EXISTS`; creation runs under
  `BEGIN EXCLUSIVE` and is idempotent (`Schema.cpp:17-62`).
- String columns are `*_id INTEGER` → `rocpd_string` (deduped) or
  `rocpd_ustring` (unique payloads). Don't store raw strings.
- Timestamps int64 ns; ids come from the table's row-id space (see
  [concepts/rpd-schema.md](../concepts/rpd-schema.md) bit layout).
- Bump the `schema_version` metadata insert **only for genuinely new
  capabilities** and be ready for feature-probe updates; readers mostly
  ignore the tag anyway.
- Generated C++ headers regenerate automatically on rebuild (`xxd -i`,
  `rpd_tracer/CMakeLists.txt:92-118`). Never hand-edit `tableSchema.h`.

## C. Writer side (if a DataSource emits rows)

Mirror an existing minimal table class: `CounterTable.cpp` or
`MonitorTable.cpp` — subclass `BufferedTable` (`Table.h:36-66`), provide a
`SqliteWriterBackend` (temp-table mode + directWrite mode, e.g.
`ApiTable.cpp:23-89`), register it in `Storage` (`Storage.cpp`) so flush /
finalize / id-offset / WAL all come for free, and add a `Logger` accessor
(`Logger.h:30-39`). If rows carry interned strings, respect the storage
generation.

## D. Reader side — update every compat surface

1. `rocpd_python`: importer/`RocpdImportData` if python tools generate the
   rows; post-processor docs.
2. `rocpd/tracing.py` if the data should reach Chrome/Perfetto export.
3. rpd_viewer: feature-probe in `util/db.py` if pages should conditionally
   show it (`:110-150`); new page per `rpd_viewer/AGENTS.md`.
4. `rpd_viewer/RPD_INFO.md` — document the table + a worked query, guarded by
   the existence-check idiom (it doubles as the chat system prompt; unlisted
   optional tables make the chat agent hallucinate).
5. Regenerate nothing else — there is no schema code in rlog or RTL that
   tracks the tracer schema (RTL writes its own subset into its own `.db`;
   if RTL must emit your table too, update `trace_db.cpp:78-159` there).

## E. Verify

`python -m rocpd.schema --create /tmp/n.rpd` (v3 path), insert probe rows,
run the new query, and open an OLD trace (e.g. repo-root `trace.rpd` or
`examples/rocm-profile-data/results/*.rpd`) in the viewer to confirm graceful
degradation. `check_skill_commands.sh`-style habit: every SQL you put in docs
must run against a real fixture.
