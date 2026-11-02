# Threading, buffers, shutdown, init order

The tracer runs inside someone else's process at arbitrary points in its
lifetime. These are the rules the code depends on; violating any of them is a
heisenbug.

## Steady-state threading model

- **One worker thread per table** (9 total) drains a fixed circular buffer in
  4096-row batches (`BufferedTable.cpp:29-41,91-122`). All queues are
  mutex+condvar — there is no lock-free queue in rpd_tracer.
- **Producer stall by design:** inserting into a full buffer blocks the
  *traced application thread* and records a `"BLOCKING"` overhead row
  (`ApiTable.cpp:138-152`) so the trace self-documents tracer-induced delay.
  Don't "fix" the blocking without a loss policy decision.
- Callbacks from GPU providers run on provider threads; flush runs on the
  flushing thread (app, autoflush worker, or signal handler via librpd_remote).
  Provider record races must use mutex + high-water mark (pattern:
  `ClrDataSource.cpp:255-296`).
- Thread-local state: string caches (`LocalStringCache.h`, `UStringCache.h`),
  pid/tid caches, marker range stacks. Marker stacks register in a global map
  keyed by (pid,tid) with RAII deregistration so the stop-time drain never
  touches destroyed TLS (`RoctxDataSource.cpp:80-99`).

## Init order (load time)

Constructor (`rpd_capi.cpp:90`) → `Logger` singleton ctor → `Logger::init()`:
Storage → factory list → dlsym/instancing → `init()` each source → autostart.

**The one hard rule:** a DataSource constructor (or anything reachable from
it) must not call `Logger::singleton()` — the static-init guard is already
held ⇒ deadlock (`Logger.h:22-26` FIXME). Mitigation pattern in-tree: defer
`singleton()` to first callback/start; Rocprof stashes its shared state
pointer (`RocprofDataSource.cpp:555-560`, comment "CRITICAL: static init").

## start/stop semantics

Ref-counted gate (`Logger.cpp:87-107`): `rpdstart` nests; sources see
`startTracing()` only on 0→1 and `stopTracing()` on 1→0 (and always at
finalize). Sources must make provider enable/disable safe to pair-call, and
**close in-flight app ranges on stop** with a stop-time timestamp (roctx
orphan-pop handling, `RoctxDataSource.cpp:195-217`), because string ids are
only valid within the current Storage generation.

## Flush

`rpdflush()` = every source `flush()` (provider → tables) → `Storage::flush()`
(tables → DB, temp→main table moves, `PRAGMA wal_checkpoint(TRUNCATE)`) →
overhead record (`Logger.cpp:109-122`, `Storage.cpp:63-76`). WAL side files
(`*.rpd-shm/-wal`) are a permanent feature; keep them with the trace file.

## Shutdown

`Logger::finalize()` (`Logger.cpp:263-290`), once-guard: stop autoflush →
`stopTracing()` all → `end()` all (deregister + final drain) → disable
overhead → `Storage::finalize()` (finalize every table, checkpoint). Ordered
before user atexit handlers via the interposition trick
([injection doc](injection-and-interposition.md)). Known fragility, from
comments: other ROCm libs crash on unload and sqlite may unload first
(`ClrDataSource.cpp:24-26` FIXME) — response is deliberate leaking
(`RocprofDataSource.cpp:463,1224` "Leak this"). Don't add shutdown-time
destruction of provider objects.

`rpd_resetStorage()` (`Logger.cpp:59-84`): park autoflush worker → delete
Storage → every source `reset()` (drop cached string ids — Storage generation
bumps, `Storage.h:40-43`) → rebuild. Any cache you add must be generation-aware
or you'll corrupt a reopened file.

## Signal safety

The tracer itself installs **no** signal handlers and its hot paths are not
async-signal-safe (std::string, fmt, malloc, json). `librpd_remote.so`
calls dlopen/dlsym from a handler — accepted but known-unsafe
(`remote/Remote.cpp:25-44`). New entry points triggered from signals must stay
as thin as Remote.cpp.

## fork()

No `pthread_atfork` anywhere: fork-without-exec leaves the child without table
worker threads. Multiprocessing must go through exec (torchrun/mpirun/
python multiprocessing default) — see
[injection doc](injection-and-interposition.md).
