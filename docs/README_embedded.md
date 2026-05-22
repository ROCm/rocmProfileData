# librpd_embedded.so — Embedded Profiling Library

`librpd_embedded.so` provides the same profiling engine as `librpd_tracer.so` but is designed for direct linking rather than `LD_PRELOAD`. It gives the application full control over initialization, configuration, and lifecycle.

## Differences from librpd_tracer.so

| | librpd_tracer.so | librpd_embedded.so |
|---|---|---|
| Usage | `LD_PRELOAD` | Link directly or `dlopen` |
| Initialization | Automatic (`__attribute__((constructor))`) | Manual via `rpdstart()` |
| Shutdown | Automatic (atexit interception) | Manual via `rpdstop()` + `rpdflush()` |
| Default autostart | 1 (tracing begins at load) | 1 (set to 0 for explicit control) |

## C API

```c
void rpd_setConfig(const char *property, const char *value);
void rpdstart();
void rpdstop();
void rpdflush();
void rpd_resetStorage();
sqlite3 *rpd_getConnection();
void rpd_rangePush(const char *domain, const char *apiName, const char *args);
void rpd_rangePop();
```

## Configuration Properties

Set properties with `rpd_setConfig()` before calling `rpdstart()`. Properties are also settable via environment variables or rlog properties. Priority order: API > environment variable > rlog property > default.

| Property | Env Var | Default | Description |
|---|---|---|---|
| `filename` | `RPDT_FILENAME` | `./trace.rpd` | Output file path, or `:memory:` for in-memory |
| `autostart` | `RPDT_AUTOSTART` | `1` | Begin tracing on init (set to `0` for embedded) |
| `autoflush` | `RPDT_AUTOFLUSH` | `0` | Periodic flush frequency in Hz (0 = off) |
| `directwrite` | `RPDT_DIRECTWRITE` | `0` | Write directly to main tables, skip temp tables |
| `delayinit` | `RPDT_DELAYINIT` | `0` | Skip singleton creation at load time |
| `stackframes` | `RPDT_STACKFRAMES` | `0` | Record call stacks |
| `datasources_explicit` | `RPDT_DATASOURCES_EXPLICIT` | (empty) | Use only these DataSources |
| `datasources_exclude` | `RPDT_DATASOURCES_EXCLUDE` | (empty) | Remove these DataSources |
| `datasources_priority` | `RPDT_DATASOURCES_PRIORITY` | (empty) | Prioritize these DataSources |
| `rocprof_noargs` | `RPDT_ROCPROF_NOARGS` | `0` | Suppress rocprofiler kernel args |

## Basic Usage

```c
#include "rpd_tracer.h"
#include <sqlite3.h>

// Configure before first use
rpd_setConfig("filename", "./my_trace.rpd");
rpd_setConfig("autostart", "0");

// Start tracing
rpdstart();

// ... application work ...

// Stop and flush
rpdstop();
rpdflush();
```

**Important:** Set `autostart` to `0` for embedded use. The default (`1`) holds its own reference on the tracing state, so a subsequent `rpdstop()` will not actually stop tracing.

## In-Memory Database

Setting `filename` to `:memory:` creates a shared in-memory database (using SQLite's memdb VFS). All table writers share a single database instance.

```c
rpd_setConfig("filename", ":memory:");
rpd_setConfig("autostart", "0");

rpdstart();

// ... profiled work ...

rpdstop();
rpdflush();

// Query results directly
sqlite3 *db = rpd_getConnection();

sqlite3_stmt *stmt;
sqlite3_prepare_v2(db, "SELECT count(*) FROM rocpd_api", -1, &stmt, NULL);
if (sqlite3_step(stmt) == SQLITE_ROW)
    printf("API calls: %d\n", sqlite3_column_int(stmt, 0));
sqlite3_finalize(stmt);

sqlite3_close(db);
```

`rpd_getConnection()` returns a new `sqlite3*` handle to the same database. The caller owns this connection and must call `sqlite3_close()` when done. Call `rpdflush()` before querying to ensure all buffered data is visible.

### Advantages of :memory:

- No disk I/O — all writes stay in RAM
- No file cleanup needed
- Query results programmatically without writing to disk
- Suitable for unit tests and benchmarks that analyze results in-process

### Drawbacks of :memory:

- Data is lost when the process exits (no persistent file)
- Not suitable for multi-process tracing (each process has its own database)
- Memory usage grows with trace size

## Controlling DataSources

DataSources are the components that capture profiling data. This includes HIP API calls, GPU kernel dispatches, and memory copies, but also user annotations (roctx/nvtx markers, rlog) and device monitoring (ROCm SMI). By default, all available DataSources are loaded.

### Available DataSources

| Name | Description |
|---|---|
| `ClrDataSource` | HIP runtime API + GPU activity via CLR profiling extensions |
| `RocprofDataSource` | ROCm profiler SDK (kernel dispatches, memory copies, HIP API) |
| `RoctracerDataSource` | ROCm tracer (legacy, HIP API + GPU activity) |
| `CuptiDataSource` | NVIDIA CUPTI (CUDA profiling) |
| `RoctxDataSource` | ROCTx user annotations (markers and ranges) |
| `NvtxDataSource` | NVTX user annotations (markers and ranges) |
| `RlogDataSource` | Remote logging annotations |
| `RocmSmiDataSource` | ROCm SMI device monitoring |

`ClrDataSource`, `RocprofDataSource`, and `RoctracerDataSource` each provide HIP API and GPU activity data. Only the highest-priority one is activated to prevent duplication. The default priority order is the order listed above.

### Disabling annotations

To capture GPU activity without user annotations:

```c
rpd_setConfig("datasources_exclude", "RoctxDataSource,NvtxDataSource,RlogDataSource");
```

### Use only specific DataSources

To capture only HIP API and GPU activity via the CLR data source:

```c
rpd_setConfig("datasources_explicit", "ClrDataSource");
```

### Choosing a HIP data provider

Use `datasources_priority` to prefer one HIP data provider over the others. The first available provider in the priority list is used:

```c
// Prefer rocprofiler-sdk over CLR and roctracer
rpd_setConfig("datasources_priority", "RocprofDataSource");
```

## Switching Files (Storage Reset)

Use `rpd_resetStorage()` to finalize the current database and open a new one. This allows splitting traces across multiple files or switching between file and in-memory modes.

```c
rpd_setConfig("filename", "./phase1.rpd");
rpd_setConfig("autostart", "0");

// Phase 1
rpdstart();
// ... work ...
rpdstop();
rpdflush();

// Switch to a new file
rpd_setConfig("filename", "./phase2.rpd");
rpd_resetStorage();

// Phase 2
rpdstart();
// ... more work ...
rpdstop();
rpdflush();
```

`rpd_resetStorage()` finalizes and closes the current database, then opens a new one using the current configuration. All cached state (string IDs, etc.) is reset. DataSources remain active across the switch.

**Sequence:** Always `rpdstop()` and `rpdflush()` before calling `rpd_resetStorage()`.

## Building

```
make
```

This builds both `librpd_tracer.so` (for LD_PRELOAD) and `librpd_embedded.so` (for direct linking).

### Linking

```
g++ myapp.cpp -L/path/to/lib -lrpd_embedded -lsqlite3 -I/path/to/include
```
