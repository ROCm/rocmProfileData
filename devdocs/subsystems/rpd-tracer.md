# rpd_tracer — the profiler library

Namespace `rpdtracer`. Everything here compiles once into `rpd_tracer_objs`
and lands in two shared libs (`rpd_tracer/CMakeLists.txt:257-281`):

- `librpd_tracer.so` — LD_PRELOAD build (`RPD_TRACER_BUILD` defined in
  `rpd_capi.cpp`): self-starting constructor + atexit interception + exported
  roctx/nvtx marker shims (`marker_shim.ver`).
- `librpd_embedded.so` — same objects, `autostart` default forced to 0
  (`rpd_capi.cpp:148-156`); app calls the 6-function C API from
  `rpd_tracer.h:9-16`: `rpdstart rpdstop rpdflush rpd_setConfig
  rpd_getConnection rpd_resetStorage` (semantics: `docs/README_embedded.md:29-41`).

Helpers: `rpdrun.c` (distributed delegate), `runTracer.sh` / `loadTracer.sh`
(launch wrappers), `rpdTracerControl.py` (python control incl. a
`sys.setprofile` python-call tracer, `:110-142`).

## Startup sequence (tracer build)

1. ELF constructor `rpdTracerInit()` (`rpd_capi.cpp:90`) → `Logger::rpdInit()`
   (`Logger.cpp:38-45`): unless `RPDT_DELAYINIT=1`, builds the `Logger`
   singleton, then `setenv("RPDT_LOADED","1")` (probed by
   `rpdTracerControl.py:82`).
2. Interposes `atexit`/`__cxa_atexit` so its finalize re-registers *after*
   every user handler → `rpdFinalize()` runs first at LIFO exit
   (`rpd_capi.cpp:94-143`). A destructor also calls finalize as backstop.
3. `Logger::init()` (`Logger.cpp:128-261`): rlog client init → create
   `Storage` (opens DB, `ensureSchema()` via `Schema.cpp:17-62`, per-PID id
   offset `sessionId<<32` at `Storage.cpp:36-45`) → build DataSource factory
   name list (default `Logger.cpp:155-165`, adjusted by the three
   `RPDT_DATASOURCES_*` knobs) → `dlsym(RTLD_DEFAULT, "<Name>Factory")` each
   (`:223`), keeping only the first of the mutually-exclusive GPU providers
   `{Clr, Rtl, Rocprof, Roctracer, Cupti}` in list order (`:200-231`) →
   `init()` all → `startTracing()` if autostart → optional autoflush thread.

Full lifecycle/invariants: [concepts/threads-and-shutdown.md](../concepts/threads-and-shutdown.md).
DataSource contract + enumeration: [concepts/datasources.md](../concepts/datasources.md).
Timestamps/correlation: [concepts/timestamps-and-clocksync.md](../concepts/timestamps-and-clocksync.md).
Schema written: [concepts/rpd-schema.md](../concepts/rpd-schema.md).

## Core objects

| Object | File | Role |
|---|---|---|
| `Logger` | `Logger.{h,cpp}` | Meyers singleton. Config, source discovery, ref-counted start/stop gate (`:87-107`), flush orchestration (`:109-122`), autoflush worker, overhead self-recording. **Reentrancy hazard: sources must not call `singleton()` from ctor paths** (`Logger.h:22-26` FIXME). |
| `Storage` | `Storage.{h,cpp}` | Owns the 9 tables + DB connection; sessionId→idOffset; annotation-id counter (bit-31 region, `Storage.h:61`); **storage generation** counter so string caches notice `resetStorage()` (`Storage.h:40-43`); WAL checkpoint per flush. |
| `BufferedTable` | `Table.h:36-66`, `BufferedTable.cpp` | Base of all 9 tables: fixed circular row buffer, **one worker thread per table**, batch size 4096, producer *blocks* when full (records a `"BLOCKING"` overhead row, e.g. `ApiTable.cpp:150`). No lock-free queue anywhere. |
| `WriterBackend` | `WriterBackend.h:10-16` | Sink abstraction: `SqliteWriterBackend` (temp table → main table on flush) or `NetWriterBackend` when this is a remote node. |
| `StringTable` / `UStringTable` | `StringTable.cpp`, `UStringTable.cpp` | `rocpd_string` is deduped (`getOrCreate`, empty string = id 1); `rocpd_ustring` is **never** deduped (args payloads). Thread-local caches: `LocalStringCache.h` (128-entry, ptr fast path), `UStringCache.h` (4096-slot FNV) — both generation-aware. |
| `DbResource` | `DbResource.{h,cpp}` | Cross-process singleton lock as a `rocpd_metadata` row + `BEGIN EXCLUSIVE` txn. Used by chronosync / remote delegate / smi sampling. |
| `Config` | `Config.cpp:23-33` | `rpd_setConfig` > env `RPDT_*` > rlog property (domain `rpd_tracer`) > default. |

Tables ↔ files: `ApiTable.cpp` (rocpd_api), `KernelApiTable.cpp`,
`CopyApiTable.cpp`, `OpTable.cpp` (rocpd_op + rocpd_api_ops),
`StringTable/UStringTable`, `MonitorTable.cpp` (RLE semantics: rows only on
value change, `:145-166`), `StackFrameTable.cpp`, `CounterTable.cpp`,
`MetadataTable.cpp` (session allocation, `:66-91`).

## Env vars (tracer knobs)

Resolution order above; "prop" = rlog property / `rpd_setConfig` key exists.

| Var | Default | Effect | Ref |
|---|---|---|---|
| `RPDT_FILENAME` | `./trace.rpd` | output; `:memory:` = shared memdb VFS | `Logger.cpp:73,134` |
| `RPDT_AUTOSTART` | 1 | trace from init; embedded/rpdrun/control force 0 | `Logger.cpp:239` |
| `RPDT_DELAYINIT` | 0 | defer Logger to first API touch | `Logger.cpp:39` |
| `RPDT_DIRECTWRITE` | 0 | skip temp tables, insert into main tables | `Logger.cpp:74,135` |
| `RPDT_AUTOFLUSH` | 0 | background flush Hz | `Logger.cpp:249-255` |
| `RPDT_STACKFRAMES` | 0 | capture stacks (needs cpptrace build) | `Logger.cpp:258` |
| `RPDT_DATASOURCES_EXPLICIT` | — | replace default source list | `Logger.cpp:141-153` |
| `RPDT_DATASOURCES_PRIORITY` | — | move/add to front (unknown names are ADDED) | `Logger.cpp:167-183` |
| `RPDT_DATASOURCES_EXCLUDE` | — | remove sources | `Logger.cpp:186-198` |
| `RPDT_ROCPROF_NOARGS` | 0 | suppress arg capture (Rocprof) | `RocprofDataSource.cpp:568` |
| `RPDT_ROCPROF_COLLECT_COUNTERS` / `_COUNTER_SETS` | 0 / 2 builtins | per-kernel HW counters | `RocprofDataSource.cpp:1328-1354`, `docs/README_gpucounters.md` |
| `RPDT_QUIET` | 0 | silence stderr | `Utility.h:106` |
| `RPDT_SMI_METRICS` / `RPDT_SMI_PERIOD` | util/sclk/temp/power, 1000µs | sysfs sampler (env-only) | `AmdSmiDataSource.cpp:167,204` |
| `RPDT_CLOCKSYNC_RANK/_MASTER/_PORT` | — | enable ChronoSync (env-only) | `Logger.cpp:208`, `ChronoSyncDataSource.cpp:188-200` |
| `RPDT_LOGAGG_HOST/_PORT`, `RPDT_NODE_ID`, `RPDT_AGENT` | — | distributed aggregation (env-only) | `Utility.h:134-157` |
| `RPDT_LOADED` | — | **written** by tracer for external detection | `Logger.cpp:44` |

## API-id / argument machinery

- `ApiIdList.{h,cpp}` — inclusion/exclusion filter over numeric API ids
  (`invertMode` XOR semantics, `ApiIdList.h:15-19`); per-provider subclasses
  build the name→id map (`RocprofApiIdList`, `RocmApiIdList`, `CudaApiIdList`).
- Argument capture is per-backend: roctracer giant `switch(cid)`
  (`RoctracerDataSource.cpp:121-732`); SDK generic arg iteration → nlohmann
  JSON (`RocprofDataSource.cpp:283-295,769-799`); CLR raw bytes hexdump,
  64-byte cap (`ClrDataSource.cpp:107-119`).
- Correlation: `ApiTable::row.api_id` doubles as the join key into
  `rocpd_op`/`rocpd_api_ops`; app annotations get ids from
  `Logger::nextAnnotationId()` (≥2^31 region).

## Counters (RocprofDataSource)

Enabled by `RPDT_ROCPROF_COLLECT_COUNTERS`; sets parsed from
`RPDT_ROCPROF_COUNTER_SETS` (`;` between sets, `,` within); attaches
`rocprofiler_configure_buffer_dispatch_counting_service` to the same lossless
buffer (`RocprofDataSource.cpp:1357-1363`); round-robins sets per kernel
*name*; results to `rocpd_counter` via `dispatchOpId` join (`:705-707,912-947`).
`counter_buffer_callback` (`:991-1095`) is dead code — never registered.

## Testing a tracer change

No unit suite covers the tracer (tests dir is clocksync-only). Standard loop:
build → `runTracer.sh -o out.rpd python3 examples/...` → `sqlite3 out.rpd
'select * from top limit 10'` → check `rocpd_metadata` `process_datasource`
row to confirm which backend was chosen (`Logger.cpp:229`).
