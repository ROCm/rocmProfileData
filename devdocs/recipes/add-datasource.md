# Recipe: add a new DataSource

Prerequisite reading: [concepts/datasources.md](../concepts/datasources.md),
[concepts/threads-and-shutdown.md](../concepts/threads-and-shutdown.md).
Templates: `RoctxDataSource.cpp` (app-callback driven), `ClrDataSource.cpp`
(provider-record driven).

1. `MyFooDataSource.{h,cpp}` in `rpd_tracer/` subclassing `rpdtracer::DataSource`,
   implementing all six virtuals (`DataSource.h:9-19`).
2. Export the C factory with the exact naming convention:
   `extern "C" { rpdtracer::DataSource *MyFooDataSourceFactory() { return new MyFooDataSource(); } }`
   (pattern `RoctxDataSource.cpp:51-53`). Discovery is pure `dlsym` —
   **a name mismatch fails silently** (this is the SMI bug; see gotchas #1).
3. Add the `.cpp` to `RPD_COMMON_SRCS` (`rpd_tracer/CMakeLists.txt:25-54`),
   or to a conditional `target_sources` block if it needs an optional
   dependency (patterns at `:179,192,200,221`). Optionally mirror into the
   legacy `rpd_tracer/Makefile` (drifting already; CMake is authoritative).
4. Register `"MyFooDataSourceFactory"` in the default list
   (`Logger.cpp:155-165`). If it's another GPU activity provider, add it to
   `rocmFactories` (`:200-206`) and choose its priority position. (Users can
   also reach an unregistered source via `RPDT_DATASOURCES_PRIORITY`, which
   *adds* unknown names — fine for bring-up, not for shipping.)
5. `init()`: acquire provider hooks, then leave quiescent (end with
   `stopTracing()`/`active=false`). Never call `Logger::singleton()` from the
   constructor path (deadlock FIXME `Logger.h:22-26`).
6. Callbacks: get tables via `Logger::singleton()` accessors
   (`Logger.h:30-39`); timestamp local events with `clocktime_ns()`, foreign
   ones with `adjust_external_ts()`; intern names with a `thread_local
   LocalStringCache` guarded by `logger.storageGeneration()`; args →
   `ustringTable().create()`; correlate with provider id into `api_id` or
   `nextAnnotationId()` for annotations.
7. `start/stopTracing()`: provider toggles must be pair-safe (refcount
   transitions only); close in-flight ranges on stop (roctx drain pattern
   `RoctxDataSource.cpp:245-257`).
8. `flush()`: move provider records into tables; serialize against provider
   callbacks with mutex + processed-count high-water mark
   (`ClrDataSource.cpp:255-296`).
9. `reset()`: clear cached string ids (called after `rpd_resetStorage()`,
   when every id from the old generation is invalid).
10. `end()`: deregister + final drain; runs before `Storage::finalize()` —
    leak provider state rather than destroying it at shutdown
    (`ClrDataSource.cpp:24-26` FIXME).
11. Config knobs via `getConfig("RPDT_MYFOO_X", "myfoo_x", default)`
    (`Config.cpp:23-33`) so env var, rlog property, and `rpd_setConfig` all
    work; add the knob to the list in `rpd_capi.cpp:52-63`,
    `docs/README_embedded.md`'s table, and
    [subsystems/rpd-tracer.md](../subsystems/rpd-tracer.md).
12. If it emits new domain/category strings, consider a domain row in
    `rpd_viewer/RPD_INFO.md` (that file is also the chat prompt — chat will
    start using your domain).

Verify: build (`cmake --build build`), run
`runTracer.sh -o /tmp/t.rpd <app>`, then
`sqlite3 /tmp/t.rpd "select * from rocpd_metadata"` — confirm your
`process_datasource` row and your tables' counts; diff overhead against a run
with `RPDT_DATASOURCES_EXCLUDE=MyFoo`.
