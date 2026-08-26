# RPD Tracer Quickstart

Build the tracer from a fresh clone of the tree, trace a workload, and switch
profiler backends.

## 1. Dependencies

ROCm must already be installed at `/opt/rocm`. Then:

```sh
# build tools not covered by install.sh
apt-get install -y build-essential cmake xxd

git clone https://github.com/ROCm/rocmProfileData.git
cd rocmProfileData
git submodule update --init rlog               # required (rlog-config, rlog annotations)
git submodule update --init rocm-trace-lite    # optional (Rtl backend)
```

## 2. Build and install

```sh
./install.sh        # installs deps, builds everything, installs to /usr/local (run as root)
```

What you get:

- `runTracer.sh` — runs any command under the tracer and writes a `.rpd` file
- `loadTracer.sh` — loads the tracer into a process without auto-start
- `rlog-config` — read/write the tracer's persistent settings
- `rpd-viewer` — web viewer for `.rpd` files

## 3. Tracing a workload

The installed `runTracer.sh` wraps your command with the tracer:

```sh
runTracer.sh -o mytrace.rpd python myworkload.py --args
```

- Output defaults to `trace.rpd` in the current directory; `-o` overrides it.
- The existing output file is removed before tracing starts.
- Subprocesses are traced automatically (the tracer is `LD_PRELOAD`ed).
- Multiple processes can append to the same `.rpd` file concurrently.

You can also attach manually:

```sh
LD_PRELOAD=librpd_tracer.so ./myworkload
```

or from inside a Python program, using the installed `rpdTracerControl` module
for start/stop and named ranges:

```python
from rpdTracerControl import rpdTracerControl
profiler = rpdTracerControl()
profiler.start()
profiler.rangePush("myDomain", "forward_pass", "batch=32")
profiler.rangePop()
profiler.stop()
```

For multi-node runs, `runTracer.sh --rank N --master <host>` sets up clock
syncing and log aggregation (see `docs/README_distributed.md`).

## 4. Switching profiler backends

Several data sources can report ROCm/HIP activity, but the tracer only ever
activates **one** of them: it walks the priority list and uses the first one
that is available. The annotation sources (roctx, NVTX, rlog, RocmSMI) run
alongside whichever backend is selected.

The GPU backends, compiled in when their libraries are present (see the
`Building with ...` lines during the build):

| Name | Source library | Notes |
|------|----------------|-------|
| `RocprofDataSource` | rocprofiler-sdk | ROCm's recommended backend; full HIP + HSA API coverage |
| `RoctracerDataSource` | libroctracer64 | classic roctracer callbacks |
| `ClrDataSource` | HIP profiler extension | HIP-internal interface |
| `RtlDataSource` | rocm-trace-lite | low-overhead dlsym interposition (needs the submodule) |
| `CuptiDataSource` | libcupti | NVIDIA systems |

Selection is controlled with three environment variables, which take the data
source class names above.

### `RPDT_DATASOURCES_PRIORITY`

Reorders the list; the first available backend wins:

```sh
RPDT_DATASOURCES_PRIORITY=RocprofDataSource runTracer.sh -o trace.rpd python myworkload.py
```

### `RPDT_DATASOURCES_EXCLUDE`

Removes data sources from the list, e.g. to turn the roctx and NVTX
annotation sources off:

```sh
RPDT_DATASOURCES_EXCLUDE=RoctxDataSource,NvtxDataSource runTracer.sh -o trace.rpd python myworkload.py
```

### `RPDT_DATASOURCES_EXPLICIT`

A whitelist: use only the listed data sources and nothing else. This example
records ClrDataSource GPU events and rlog annotations, with no other backends
or annotation messages:

```sh
RPDT_DATASOURCES_EXPLICIT=ClrDataSource,RlogDataSource runTracer.sh -o trace.rpd python myworkload.py
```

## 5. Environment variable options

Every `RPDT_*` variable can be overridden by a persistent property of the same
name (see [rlog-config](#6-persistent-settings-with-rlog-config)); the
precedence is: programmatic `setConfig()` call > environment variable > rlog
property > built-in default.

| Variable | Default | Effect |
|----------|---------|--------|
| `RPDT_FILENAME` | `trace.rpd` | Output file path |
| `RPDT_AUTOSTART` | `1` | Set `0` to load the tracer without recording (manual start/stop) |
| `RPDT_AUTOFLUSH` | `0` | Auto-flush frequency in Hz |
| `RPDT_DIRECTWRITE` | `0` | Bypass buffering, write straight to the DB |
| `RPDT_DELAYINIT` | `0` | Defer tracer init until first use (set by `loadTracer.sh`) |
| `RPDT_DATASOURCES_EXPLICIT` | — | Only use the listed data sources |
| `RPDT_DATASOURCES_PRIORITY` | — | Reorder data source priority |
| `RPDT_DATASOURCES_EXCLUDE` | — | Remove listed data sources |
| `RPDT_STACKFRAMES` | `0` | Capture host stack frames |
| `RPDT_ROCPROF_NOARGS` | `0` | Set `1` to skip recording API arguments (Rocprof backend) |
| `RPDT_ROCPROF_COLLECT_COUNTERS` | `0` | Collect GPU hardware counters (Rocprof backend) |
| `RPDT_ROCPROF_COUNTER_SETS` | — | Counter set names for the above |
| `RPDT_CLOCKSYNC_RANK` / `RPDT_CLOCKSYNC_MASTER` | — | Multi-node clock syncing (set via `--rank` / `--master`) |
| `RPDT_LOGAGG_HOST` / `RPDT_LOGAGG_PORT` | — | Multi-node log aggregation (set via `--master`) |

## 6. Persistent settings with rlog-config

Instead of exporting `RPDT_*` variables for every run, values can be stored
persistently in a per-user property database (`$HOME/.rlog.db`). On startup the
tracer reads any property it knows about from the `rpd_tracer` domain, so a
value set once applies to every later run.

```sh
rlog-config                                  # list all stored properties
rlog-config get rpd_tracer:filename          # show one value
rlog-config set rpd_tracer:filename /tmp/mytrace.rpd
```

`runTracer.sh` consults the properties for `filename`, `clocksync_port`,
`logagg_port`, and `exit_delay`. So:

```sh
rlog-config set rpd_tracer:filename /data/traces/mytrace.rpd
runTracer.sh python myworkload.py            # writes to /data/traces/mytrace.rpd
```

Note that an environment variable, if set, wins over the stored property.

## 7. Quick check: count api and op rows

An `.rpd` file is a SQLite database. The CPU-side API calls live in
`rocpd_api` and the GPU-side operations in `rocpd_op`:

```python
import sqlite3

con = sqlite3.connect("mytrace.rpd")
api_rows = con.execute("select count(*) from rocpd_api").fetchone()[0]
op_rows  = con.execute("select count(*) from rocpd_op").fetchone()[0]
print(f"api rows: {api_rows}")
print(f"op rows:  {op_rows}")
```

(Or interactively: `sqlite3 mytrace.rpd` then `select count(*) from rocpd_api;`)

## 8. rpd_viewer

`rpd_viewer` is a web-based viewer for `.rpd` files: a Dash app that opens the
trace in a browser so you can explore kernels, API calls, timelines, counters,
and more without exporting JSON to Perfetto.

```sh
cd rpd_viewer
make install                    # pip install . -> installs the rpd-viewer command
rpd-viewer mytrace.rpd          # or just: rpd-viewer  (picks the file in the browser)
```

Then open `http://localhost:8050` (use `--host` / `--port` to change the
binding). Pages include a summary Dashboard, kernel/API/op/copy tables, a
Timeline page that opens the trace in Perfetto, GPU monitor charts, and a
read-only SQL query console. If the trace file was modified within the last
30 seconds the Dashboard auto-refreshes, treating it as a live session.

## 9. Quick hack: analyze a trace with an LLM agent

The fastest way to answer an ad-hoc question about a trace:

1. Start any LLM coding agent (Claude Code, opencode, etc.) in the repo.
2. Point it at `rpd_viewer/RPD_INFO.md` — the RPD schema and analysis
   reference (tables, views, units, common pitfalls).
3. Ask your questions about the `.rpd` file (it is just SQLite); the agent can
   query the file directly and iterate.
