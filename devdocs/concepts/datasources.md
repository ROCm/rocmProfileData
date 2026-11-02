# DataSources

Contract is six pure virtuals (`rpd_tracer/DataSource.h:9-19`) — no configure
hooks, config is read inside the source via `getConfig()`:

```cpp
virtual void init() = 0;          // register hooks, stay QUIESCENT after
virtual void startTracing() = 0;  // called only on refcount 0→1
virtual void stopTracing() = 0;   // refcount 1→0 AND at finalize; close in-flight ranges
virtual void flush() = 0;         // drain provider records into Logger tables
virtual void end() = 0;           // deregister + final drain (before Storage::finalize)
virtual void reset() {}           // invalidate cached string ids after rpd_resetStorage()
```

Discovery: sources export `extern "C" DataSource* <Name>DataSourceFactory()`;
`Logger::init()` iterates a name list and `dlsym`s each (`Logger.cpp:223`) —
missing symbol = silently skipped (only if built without it, so a typo'd name
fails silently too; see SMI bug in [gotchas.md](../gotchas.md)).

Selection: default order `Logger.cpp:155-165`; `RPDT_DATASOURCES_EXPLICIT`
replaces the list; `_PRIORITY` moves/adds to front (unknown names are added —
that's the workaround for unregistered sources); `_EXCLUDE` removes.
**Exactly one GPU provider survives**: first match of `{Rocprof, Clr,
Roctracer, Rtl, Cupti}` (`:200-231`). Effective default priority (whatever was
compiled in): **Clr > Rtl > Rocprof > Roctracer > Cupti** — with a modern ROCm
install that has `hip/amd_detail/hip_profiler_ext.h`, CLR wins;
`docs/README_tracer_quickstart.md:79` claiming Rocprof is the default is
stale. Chosen source recorded as `process_datasource` metadata row
(`Logger.cpp:229`) — check it first when debugging "why no kernels?".

## Inventory

| Source | Provider / mechanism | Tables filled | Gated by |
|---|---|---|---|
| `ClrDataSource` | HIP runtime CLR profiling ext: chunk callback + `hipProfilerGetRecordsExt` slab iteration; API+GPU fused records. **Slab lifetime invariant: callback and flush share `m_flushMutex` + `m_processedCount` high-water mark — the provider frees the slab when the callback returns** (`ClrDataSource.h:61-65`) | api, op+api_ops, kernelapi, copyapi | built iff `hip_profiler_ext.h` found |
| `RocprofDataSource` | rocprofiler-sdk: exported `rocprofiler_configure` handshake (`:1171-1188`), one LOSSLESS 0x40000 buffer + callback thread; buffer-tracing services KERNEL_DISPATCH / MEMORY_COPY / HIP_RUNTIME_API_EXT / SCRATCH_MEMORY / CORRELATION_ID_RETIREMENT / KFD_*; code-object cb demangles kernel names; counters via dispatch counting service | api, op, kernelapi, copyapi, counter | SDK found AND ROCm major ≠ 6 |
| `RoctracerDataSource` | legacy roctracer enter/exit callbacks + HCC ops pool; giant per-cid arg switch; mandatory default pool "or crash at exit" (`:897-904`); only source using `Unwind.cpp` stack frames | api, op, kernelapi, copyapi | roctracer64 found |
| `RtlDataSource` | bridge into embedded rocm-trace-lite sources; start/stop/flush are no-ops; forces `HSA_TOOLS_LIB` + legacy tool-load path | api, op, kernelapi, copyapi | `RPD_ENABLE_ROCM_TRACE_LITE=ON` |
| `CuptiDataSource` | NVIDIA CUPTI subscribe + activity buffers; converts CUPTI CLOCK_REALTIME by sampling both clocks (approximate hack, `:652-658`) | api, op, kernelapi, copyapi | CUDA found |
| `RoctxDataSource` / `NvtxDataSource` | **implements** the marker APIs itself (LD_PRELOAD interposition + `marker_shim.ver`); per-thread range stacks registered for shutdown drain; orphan pops get `<pre-existing>` placeholders clamped to resume time (`RoctxDataSource.cpp:195-217`) | api (roctx/nvtx domain) | always |
| `RlogDataSource` | registers as `rlog::Logger` on the hub at startTracing; marks/ranges with arbitrary domain/category → api rows; owns its stack + drain-on-stop | api | always |
| `AmdSmiDataSource` | polls `/sys/class/drm/cardN/device/gpu_metrics` (no libamd_smi despite the name); worker thread; RLE writes to monitor table; `DbResource("amdsmi_logger_active")` cross-process singleton | monitor | built, but **unreachable by default — factory-name mismatch bug**, see gotchas |
| `ChronoSyncDataSource` | clock-sync service (see [timestamps-and-clocksync.md](timestamps-and-clocksync.md)) | metadata (+shm) | `RPDT_CLOCKSYNC_RANK` |
| `RemoteDataSource` | node-0 TCP receiver for multi-node | all 9 tables | `RPDT_LOGAGG_PORT` + node 0 |

## Writing callbacks (rules)

- Timestamps: local events `clocktime_ns()`; foreign-clock records
  `adjust_external_ts()` — see [timestamps-and-clocksync.md](timestamps-and-clocksync.md).
- Strings: `thread_local LocalStringCache` + check `logger.storageGeneration()`;
  args payloads → `ustringTable().create()` (never deduped).
- Correlate GPU work to the launching API row via provider correlation id in
  `api_id`; app-level ranges get `logger.nextAnnotationId()`.
- `init()` must end quiescent and **never call `Logger::singleton()` from the
  constructor** (`Logger.h:22-26` deadlock FIXME).
- Guard flush-vs-callback races (copy `ClrDataSource::flush`, `:255-296`).

To add a new source end-to-end: [recipes/add-datasource.md](../recipes/add-datasource.md).
