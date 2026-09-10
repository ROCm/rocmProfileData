# rlog Benchmarks

Standalone executables for measuring `rangePush`/`rangePop` throughput.
Build them from the `build/` directory:

```
cmake .. && make bench_loopback bench_external bench_roctx bench_guard bench_range
```

A Python equivalent lives in `../python/bench_range.py`.

---

## bench_loopback

Measures rlog's **internal dispatch overhead** with no external tool involved.

A no-op `Logger` is registered with the Hub before the timed loop, so every
`rangePush`/`rangePop` call traverses the full client → Hub → Logger path but
does no real work in the Logger itself.  This establishes the baseline cost of
using rlog in a instrumented application.

```
./bench_loopback
loopback: 1000000 iterations  109.5 ms  (110 ns/pair  9.13 Mpairs/sec)
```

---

## bench_external

Measures **end-to-end throughput when a profiling tool is attached via rlog**.

No `Logger` is registered by this binary.  When run standalone the Hub has no
loggers, `isActive()` returns false, and the calls are no-ops — this gives a
useful no-tool baseline for comparison.

One million pairs are issued in a single continuous run.  Two timing windows
are reported from within that run:

**burst** — 10 000 pairs measured after a 1000-pair warmup, while the tool's
buffers are still empty.  Reflects peak throughput before any buffer pressure
builds up.

**sustained** — the full 1 000 000 pairs.  Reflects steady-state throughput
once the tool is under continuous load.

A tool that is faster on burst than sustained is relying on buffering and
cannot keep up under continuous load.  Both numbers should be compared against
the no-tool baseline to isolate the tool's own overhead.

```
./bench_external
external: tool active = no (baseline)

burst:       10000 pairs       1.6 ms  (  163 ns/pair  6.14 Mpairs/sec)  [after 1000 warmup]
sustained: 1000000 pairs     163.1 ms  (  163 ns/pair  6.13 Mpairs/sec)
```

Run under a tool to measure how fast it can service the requests:

```
# rpd_tracer
loadTracer.sh ./bench_external

# rocprofv3
rocprofv3 --sys-trace ./bench_external
```

---

## bench_roctx

Same sustained + burst measurement as `bench_external` but dispatches through
roctx (`libroctx64.so`) instead of the rlog Hub.  rlog is explicitly disabled
so only the roctx path is exercised.  Skips with a message if `libroctx64.so`
is not installed.

```
./bench_roctx
roctx: tool active = no (baseline)

burst:       10000 pairs       1.1 ms  (  114 ns/pair  8.78 Mpairs/sec)  [after 1000 warmup]
sustained: 1000000 pairs     114.3 ms  (  114 ns/pair  8.75 Mpairs/sec)
```

Run under a tool:

```
loadTracer.sh ./bench_roctx
rocprofv3 --sys-trace ./bench_roctx
```

---

## bench_guard

Measures the cost of **guarding calls with a cached `isLogging` flag** when no
tool is attached.  `isLogging` is an `std::atomic<bool>` kept in sync with the
Hub's active state via a registered callback.  When `isLogging` is false the
`rangePush`/`rangePop` calls are never made.

```
./bench_guard
guard: isLogging = false (baseline)

burst:       10000 pairs       0.0 ms  (    5 ns/pair  204.95 Mpairs/sec)  [after 1000 warmup]
sustained: 1000000 pairs       4.9 ms  (    5 ns/pair  204.78 Mpairs/sec)
```

This result (~5 ns/pair) is representative of a **properly designed production
application**: instrumented with rlog, ready to log, but with no tool currently
attached.  The only runtime cost is a single atomic bool load and a
correctly-predicted not-taken branch per call site.

Compare this against `bench_roctx` (~114 ns/pair baseline, tool not attached).
Libraries that unconditionally call roctx pay that cost in production whether a
profiler is running or not.  With rlog and a guard, the production overhead is
~23x lower, and drops to zero if the call sites are omitted entirely when
`isLogging` is false.

Note that every call site here passes string literals, which cost nothing to
pass.  Real call sites rarely do — see `bench_range`.

---

## bench_range

Measures `rlog::Range` when the range arguments are **expensive to build**.
This is the case `bench_guard` does not cover: a guard only protects the calls
it wraps, not the arguments those calls are given.

Every variant formats the same five-field string.  The only difference is when
the formatting happens.  Passing the string eagerly evaluates it at the call
site, *before* the constructor is entered, so the guard inside the constructor
is already too late.  Passing a lambda defers it until after the guard passes.

```
./bench_range
range: logging = false (baseline)

no tool attached:
  empty loop                        5.5 ms  (      5 ns/range)
  Range, literal args              16.2 ms  (     16 ns/range)
  Range, eager fmt args           520.6 ms  (    521 ns/range)
  Range, deferred lambda           17.0 ms  (     17 ns/range)

no-op Logger attached (logging = true):
  Range, literal args              74.3 ms  (     74 ns/range)
  Range, eager fmt args           604.9 ms  (    605 ns/range)
  Range, deferred lambda          607.7 ms  (    608 ns/range)
```

Two results matter:

**With no tool attached, eager arguments cost ~521 ns per range** — about 30x
the guarded literal case, and paid on every call in production.  The deferred
lambda costs ~17 ns, indistinguishable from passing a literal.  The formatting
simply never runs.

**With a tool attached both cost the same** (~605 ns): the work has to be done
either way, and the lambda adds nothing.  There is no tradeoff to weigh — the
deferred form is never slower, so expensive range arguments should always be
passed as a callable.
