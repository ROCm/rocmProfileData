# rlog Tests

Run all tests from the `build/` directory:

```
cmake .. && make && ctest
```

---

## mock_roctx.so

Not a test — a shared library built for use by other tests.  Implements the
three roctx symbols that `rlog.cpp` loads via `dlopen`:

- `roctxMarkA`
- `roctxRangePushA`
- `roctxRangePop`

Each function increments a corresponding `int` counter exported from the
library (`mock_roctx_mark_count`, `mock_roctx_push_count`,
`mock_roctx_pop_count`).  Tests open the mock with `RTLD_NOLOAD` and read
these counters via `dlsym` to verify that dispatch actually reached the library.

---

## env_vars (`test_env.cpp`)

Sets `RLOG_FORCE_ROCTX=1` and `RLOG_ROCTX_LIBPATH=<mock_roctx.so>`, then
calls `rlog::init()`.

Verifies:
- `rlog::enabled(Roctx)` is true
- `rlog::isActive()` is true
- `mock_roctx.so` was the library actually opened (confirmed via `RTLD_NOLOAD`)
- `mark`, `rangePush`, and `rangePop` each increment the mock's call counters
  by exactly 1

This test confirms that `RLOG_FORCE_ROCTX` triggers loading, that
`RLOG_ROCTX_LIBPATH` correctly overrides the default library path, and that
all three dispatch functions reach the loaded library.

---

## libpath_no_force (`test_libpath_no_force.cpp`)

Sets `RLOG_ROCTX_LIBPATH=<mock_roctx.so>` but does **not** set
`RLOG_FORCE_ROCTX`, then calls `rlog::init()`.

Verifies:
- `rlog::enabled(Roctx)` is false

Confirms that `RLOG_ROCTX_LIBPATH` alone is not sufficient to enable roctx —
the force flag or an explicit `setEnabled(Roctx, true)` call is also required.

Runs in a separate binary from `env_vars` so the static globals in `rlog.cpp`
start zeroed.

---

## roctx_real (`test_roctx_real.cpp`)

Sets `RLOG_FORCE_ROCTX=1` (no `RLOG_ROCTX_LIBPATH`) and calls `rlog::init()`,
using the default library name `librocprofiler-sdk-roctx.so`.

**Skips** (exit code 77, reported as `Skipped` by ctest) if the library is not
found on the system.

Verifies when the library is present:
- `rlog::enabled(Roctx)` is true
- `rlog::isActive()` is true
- `mark`, `rangePush`, and `rangePop` complete without crashing

Confirms that the default library name and symbol names are correct for the
real roctx installation.

---

## nvtx_real (`test_nvtx_real.cpp`)

Sets `RLOG_FORCE_NVTX=1` (no `RLOG_NVTX_LIBPATH`) and calls `rlog::init()`,
using the default library name `libcupti.so`.

**Skips** if `libcupti.so` is not found (requires a CUDA installation).

Verifies when the library is present:
- `rlog::enabled(Nvtx)` is true
- `rlog::isActive()` is true
- `mark`, `rangePush`, and `rangePop` complete without crashing

Confirms that the default NVTX library name and symbol names are correct for
the real CUPTI installation.
