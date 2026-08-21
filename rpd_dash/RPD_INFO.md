# RPD Trace Schema & Analysis Reference

RPD (ROCm Profile Data) files are SQLite databases. You query them read-only
(`PRAGMA query_only=ON`), with a **10 s timeout**, **500 row cap**, and results
truncated at 6000 chars. Prefer aggregates over raw dumps. Never write, `CREATE`,
or `ATTACH`.

**Part I is invariant** — true of any `.rpd`. **Part II is observed** in one
reference trace and must be confirmed by a discovery query before you rely on it.

---

# PART I — INVARIANT SCHEMA

## 1. Units and time

| Thing | Unit |
|---|---|
| `start`, `end` on every table | **nanoseconds**, int64 |
| Duration | `end - start` ns |
| Convert | `/1e3` → µs, `/1e6` → ms, `/1e9` → s |
| `rocpd_counter.value` | REAL, counter-defined |

Timestamps are a monotonic clock (values near 3.3e16 are normal); they are **not**
wall-clock epoch. Only differences are meaningful.

Use float division (`/1e6`), not `/1000000` — integer division silently truncates,
which is why some shipped queries report `0` for sub-millisecond work.

**Duration can be negative.** GPU op timestamps come from hardware and are not
guaranteed ordered. Guard aggregates with `WHERE end > start` when computing
bandwidth or rates.

```sql
SELECT COUNT(*) FROM rocpd_op WHERE end < start;
-- reference trace: 20  (all SDMA Memcpy ops, down to -909 µs)
```

## 2. String interning

Nothing stores text inline. `rocpd_api` and `rocpd_op` hold `*_id` integers
pointing at two string tables:

- **`rocpd_string`** — short, highly repeated: domains, categories, API names,
  kernel names, op types.
- **`rocpd_ustring`** — "unique strings": the `args` payload only. Separate
  because args have high cardinality and are never deduped.

`rocpd_api.args_id` → `rocpd_ustring`. **Every other `*_id` → `rocpd_string`.**
Getting this backwards is the single most common schema error.

The built-in views resolve all of this. **Prefer the views** (`api`, `op`,
`kernel`, `copyop`) unless you need speed on a hot path.

## 3. Tables

| Table | Rows are | Key columns |
|---|---|---|
| `rocpd_api` | one CPU-side call/range | `id, pid, tid, start, end, apiName_id, category_id, domain_id, args_id` |
| `rocpd_op` | one GPU-side execution | `id, gpuId, queueId, sequenceId, start, end, description_id, opType_id` |
| `rocpd_api_ops` | the CPU→GPU bridge | `api_id, op_id` |
| `rocpd_kernelapi` | launch config, 1:1 with a launch api | `api_ptr_id, stream, gridX/Y/Z, workgroupX/Y/Z, groupSegmentSize, privateSegmentSize, kernelName_id` |
| `rocpd_copyapi` | copy params, 1:1 with a copy api | `api_ptr_id, stream, size, kind, dst, src, dstDevice, srcDevice, sync, pinned` |
| `rocpd_string` / `rocpd_ustring` | interned text | `id, string` |
| `rocpd_metadata` | tags | `tag, value` |
| `rocpd_monitor` | SMI samples (power/temp/clock) | `deviceType, deviceId, monitorType, start, end, value` |
| `rocpd_counter` | per-dispatch HW counters | `op_id, name_id, value` |
| `rocpd_stackframe` | host callstack frames | `api_ptr_id, depth, name_id` |

`rocpd_monitor`, `rocpd_counter`, `rocpd_stackframe` are **frequently empty** —
they require the tracer to be configured for them. Always count before analyzing.

### `id` encoding — sessions

`id` values are offset per capture session: `offset = session_id << 32`
(`rpd_tracer/Storage.cpp:36`). So ids are huge and **sparse**, and `id >> 32`
recovers the session — which in a multi-process run is effectively the rank.

```sql
SELECT id>>32 AS session, pid, COUNT(*) n FROM rocpd_api GROUP BY 1,2;
-- reference trace:
-- 0|330960|1        1|331100|354997        2|331101|355229
```
Never assume `id` is dense, ordered across processes, or usable as a row number.

## 4. The `rocpd_api` → `rocpd_api_ops` → `rocpd_op` bridge

This is the central relationship: a **CPU dispatch** (`rocpd_api`) linked to the
**GPU execution** it caused (`rocpd_op`). Traverse it in both directions.

```
rocpd_api.id  ──<  rocpd_api_ops.api_id / .op_id  >──  rocpd_op.id
      │                                                     │
 rocpd_kernelapi.api_ptr_id                          description_id → kernel name
 rocpd_copyapi.api_ptr_id                            opType_id → KernelExecution|Memcpy|Barrier
```

**CPU → GPU** (what did this call run?):
```sql
SELECT C.string AS apiName, COUNT(*) n, SUM(O.end-O.start)/1e6 AS gpu_ms
FROM rocpd_api_ops X
JOIN rocpd_api A ON A.id = X.api_id
JOIN rocpd_op  O ON O.id = X.op_id
JOIN rocpd_string C ON C.id = A.apiName_id
GROUP BY 1 ORDER BY gpu_ms DESC;
```
```
hipStreamWaitEvent|1020|5237.75        hipExtModuleLaunchKernel|40480|4713.09
hipLaunchKernel|29760|1299.84          hipModuleLaunchKernel|564|890.50
hipMemcpyAsync|26614|114.03            hipMemcpyWithStream|654|7.34
```
(0.1 s.) Note `hipStreamWaitEvent` tops the list only because it produces
`Barrier` ops — that is *waiting*, not work. See §7.

**GPU → CPU** (who launched this kernel?): same join, filter on `rocpd_op`.

Only a minority of APIs produce ops. In the reference trace 99,146 of 710,227
APIs do. The bridge is 1:1 there, but **do not assume it** — one api can map to
several ops (graph launches especially). Use `COUNT(DISTINCT api_id)` when
counting calls.

### Dispatch latency
```sql
SELECT C.string AS apiName, COUNT(*) n, AVG(O.start - A.end)/1e3 AS avg_gap_us
FROM rocpd_api_ops X
JOIN rocpd_api A ON A.id=X.api_id
JOIN rocpd_op  O ON O.id=X.op_id
JOIN rocpd_string C ON C.id=A.apiName_id
GROUP BY 1 ORDER BY n DESC LIMIT 5;
```
The gap is often **negative** (async launch returns after the GPU already
started) and, for queued work, can be enormous — it measures backlog, not
overhead. Only meaningful when the queue is empty; filter with the delta-scan
in §9 (`depth = 1`) for true launch latency.

## 5. Built-in views (always present)

Real definitions, with the semantics that matter:

| View | What it is | Watch out |
|---|---|---|
| `api` | `rocpd_api` + resolved `domain, category, apiName, args` | the workhorse |
| `op` | `rocpd_op` + resolved `description, opType` | |
| `kernel` | ops joined to `rocpd_kernelapi`: adds `duration`, grid/workgroup, `kernelName` | **kernels only** |
| `copy` | `rocpd_copyapi` + `rocpd_api` — **API timing** | contains non-copies, see §11 |
| `copyop` | copies joined through to `rocpd_op` — **GPU timing** | use this for bandwidth |
| `top` | ops ranked by total duration, with `Percentage` | includes `Barrier`, see §7 |
| `busy` | per-GPU `GpuTime/WallTime` | **not a 0–1 fraction**, see §7 |
| `stackframe` | host callstack frames | empty unless collected |
| `napi`, `nop` | multi-node decoders: split `pid`/`gpuId` by the stride metadata into `node` + local id | only for merged multi-node traces |

`top` picks the op name intelligently: `description` when non-empty, else
`opType`. That is why memcpys show up as `Memcpy` and kernels by name.

**Version-dependent views.** `counter` and `counter_summary` (per-dispatch and
aggregated HW counters) are defined in recent `utilitySchema.cmd` but are
**absent from older traces** — the reference trace does not have them. Querying
one that is missing fails with `no such table`, it does not return zero rows.
Probe before use (§14). The underlying `rocpd_counter` table is always present,
so `SELECT COUNT(*) FROM rocpd_counter` is always safe.

## 6. The domain taxonomy

`rocpd_api` rows carry `(domain, category, apiName, args)`. Domains are of four
kinds:

| Kind | Domains | Meaning |
|---|---|---|
| **Driver** | `hip`, `cuda` | CPU-side calls dispatching/managing GPU work. The host→device bridge. **Not application semantics.** |
| **Marker / annotation** | `roctx`, `nvtx` | User ranges and marks. Conventionally `apiName='UserMarker'`, `category` of `range` or `mark`; **the label lives in `args`**. |
| **Higher-order (rlog)** | caller-defined — e.g. `torch`, `miopen` | Library/framework instrumentation via the rlog API. |
| **Profiler self-instrumentation** | `rpd_tracer` | The tracer measuring **its own overhead** (`category='overhead'`). Never report this as application activity. |

In `rpd_tracer/RlogDataSource.cpp` (`mark()` at :140, `rangePush()` at :159),
`domain`, `category`, and `apiName` are **arbitrary caller-supplied strings**
interned at call time. That is how `torch` and `miopen` appear with no
roctx/nvtx involvement. **The set of domains is open-ended and unpredictable.**

Note the structural difference: roctx/nvtx put the label in **`args`** with a
fixed `apiName='UserMarker'` (`RoctxDataSource.cpp:96`), while rlog domains put
the label in **`apiName`** and use `args` for structured data. A trace may use
either convention.

### Standing first move on any unfamiliar trace
```sql
SELECT domain, category, COUNT(*) AS calls
FROM api GROUP BY domain, category ORDER BY calls DESC;
```
(0.8 s on 710 k rows.) Reference trace:
```
torch|function|334290          miopen|function|211792
hip||114577                    torch|backward_function|27120
miopen|driver|21256            torch|user_scope|884
rpd_tracer|overhead|308
```

**The `hip` domain uses an empty-string category (`''`), never NULL.** Verified:
114,577 rows, 0 NULL, 114,577 empty. Any `category` filter or `GROUP BY` must
account for it — `WHERE category IS NOT NULL` does **not** exclude it.

Zero-duration rows (`start = end`) are **marks**; non-zero are **ranges**.
Reference trace: all 21,256 `miopen/driver` rows are marks.

## 7. Sentinels and surprising semantics

**`busy.Busy` is not a fraction.** Observed > 1.0. `WallTime` is global across
all GPUs while `GpuTime` sums *overlapping* ops on multiple queues.
```sql
SELECT * FROM busy;
-- gpuId 2: GpuTime 5972214536  WallTime 5960672000  Busy 1.0019
-- gpuId 3: GpuTime 6290876016  WallTime 5960672000  Busy 1.0554
```
Correct per-GPU formulation — own span as the denominator, and exclude `Barrier`
(which is wait time, not work):
```sql
SELECT gpuId,
       SUM(end-start)/1e6 AS gpu_ms,
       (MAX(end)-MIN(start))/1e6 AS span_ms,
       100.0*SUM(end-start)/(MAX(end)-MIN(start)) AS busy_pct
FROM rocpd_op
WHERE opType_id NOT IN (SELECT id FROM rocpd_string WHERE string='Barrier')
GROUP BY gpuId;
-- 2|3408.7|5957.7|57.2      3|3616.7|5960.6|60.7
```
Even this can exceed 100% when a GPU has concurrent queues. For a true
occupancy figure use the time-weighted delta-scan in §9.

**`Barrier` ops dominate naive totals.** In the reference trace `Barrier` is
42.7% of `SUM(end-start)` over `rocpd_op` — but it is `hipStreamWaitEvent`
*waiting*, not work. `top` and `busy` both include it. Exclude it from
utilization and "top kernel" answers.

**`rocpd_op.description` is empty for most non-kernels.** Kernels always have a
name; memcpys usually have `''` (26,570) with a few tagged `SDMA` (698) or
`Fill` (54); barriers always `''`. Group on `opType` for those.

**Kernel-name form is not guaranteed.** Whether a name is C++-mangled
(`_Z23ncclDevKernel...`) or plain (`MIOpenBatchNormBwdSpatial`) depends on the
data source that captured it: `RoctracerDataSource.cpp:137` and
`CuptiDataSource.cpp:192` demangle via `cxx_demangle()`, while
`ClrDataSource.cpp:164` stores the raw name. **Both forms routinely coexist in
one trace** — the reference trace has 24 mangled names (30,324 launches)
alongside 51 plain ones (40,480). Never assume a form; if you need to match a
kernel by name, check first:
```sql
SELECT CASE WHEN description LIKE '\_Z%' ESCAPE '\' THEN 'mangled' ELSE 'plain' END AS form,
       COUNT(*) AS distinct_names, SUM(n) AS launches
FROM (SELECT description, COUNT(*) n FROM op WHERE opType='KernelExecution' GROUP BY 1)
GROUP BY 1;
-- reference trace: mangled|24|30324    plain|51|40480
```
Prefer `LIKE '%substring%'` over equality, and expect that a demangled name and
its mangled counterpart will not match each other.

**`$.seq` uses `-1` as "not part of an autograd sequence".** Any pairing or
grouping must filter `>= 0` or results silently collapse onto one bogus key.

**`json_extract(args, ...)` HARD-ERRORS on non-JSON args** — the whole query
fails with `malformed JSON`, it does not return NULL. `args` is plain text for
`hip` (empty), `miopen/driver` (a command line), and `rpd_tracer` (`k=v | k=v`).
**Always guard**:
```sql
WHERE json_valid(args) AND json_extract(args,'$.seq') >= 0
```
`args LIKE '{%'` also works and is slightly faster. This is mandatory, not
defensive.

**Nested CPU time double-counts.** `rocpd_api` ranges nest, so
`SUM(end-start)` over APIs (39,475 ms) wildly exceeds the wall span (11,399 ms).
For attributable CPU time use exclusive time from `ext_callstack` (§10).

## 8. `args` as JSON

Where `args` is JSON, use `json_extract`. Useful idioms:

```sql
-- keys present in a domain's args
SELECT DISTINCT key FROM api, json_each(api.args)
WHERE domain='torch' AND json_valid(args) LIMIT 20;

-- group by an extracted field
SELECT json_extract(args,'$.sizes') AS sizes, COUNT(*) n
FROM api WHERE domain='torch' AND apiName='aten::conv2d' AND json_valid(args)
GROUP BY 1 ORDER BY n DESC LIMIT 5;
```
This is the general mechanism — **correlate annotations via `json_extract` on
`args`** — that the Part II patterns are instances of.

## 9. Queue depth, utilization, idle — the delta-scan

**Do not write a correlated self-join over ops.** It does not finish (killed at
60 s on the reference trace). Use the linear delta-scan from `tracing.py`:
emit `+1` at each start and `-1` at each end, order by timestamp, take a running
sum.

```sql
WITH ev AS (
  SELECT gpuId, start AS ts,  1 AS d FROM rocpd_op
  UNION ALL
  SELECT gpuId, end   AS ts, -1 AS d FROM rocpd_op
), run AS (
  SELECT gpuId, ts,
         SUM(d) OVER (PARTITION BY gpuId ORDER BY ts, d DESC
                      ROWS UNBOUNDED PRECEDING) AS depth,
         LEAD(ts) OVER (PARTITION BY gpuId ORDER BY ts, d DESC) AS nxt
  FROM ev
)
SELECT gpuId,
       SUM(depth*(nxt-ts))*1.0/SUM(nxt-ts) AS mean_depth,
       100.0*SUM(CASE WHEN depth>0 THEN nxt-ts ELSE 0 END)/SUM(nxt-ts) AS pct_busy
FROM run WHERE nxt IS NOT NULL
GROUP BY gpuId;
-- 2|1.0024|54.85        3|1.0554|56.70      (0.6 s)
```
`ORDER BY ts, d DESC` puts starts before ends at equal timestamps so
zero-length and coincident ops nest rather than cross.

**Idle gaps** — same scan, `depth = 0`:
```sql
WITH ev AS (SELECT gpuId, start ts, 1 d FROM rocpd_op
            UNION ALL SELECT gpuId, end, -1 FROM rocpd_op),
run AS (SELECT gpuId, ts,
        SUM(d) OVER (PARTITION BY gpuId ORDER BY ts, d DESC ROWS UNBOUNDED PRECEDING) depth,
        LEAD(ts) OVER (PARTITION BY gpuId ORDER BY ts, d DESC) nxt FROM ev)
SELECT gpuId, SUM(nxt-ts)/1e6 AS idle_ms, COUNT(*) AS gaps,
       MAX(nxt-ts)/1e6 AS longest_gap_ms
FROM run WHERE depth=0 AND nxt IS NOT NULL AND nxt>ts GROUP BY gpuId;
-- 2|2684.5|3040|896.9      3|2578.7|2647|939.7      (0.5 s)
```
`tracing.py` builds queue depth slightly differently (`+1` at **api** start,
`-1` at **op** end) to model submitted-but-not-complete work. Use that variant
when the question is about queueing rather than GPU occupancy. Note it can go
negative at a window edge if a `-1` is clipped from its `+1`.

## 10. Call hierarchy — `ext_callstack` (optional table)

When present, this is **the** correct way to attribute GPU work to an
application-level annotation.

`ext_callstack` is an **ancestor-closure** table, not an edge list. For every
frame it emits one row per ancestor:
`(parent_id, child_id, depth, cpu_time, gpu_time)`.

- `depth` is **relative** — distance from `parent_id` down to `child_id`.
- `depth = 0` ⇒ `parent_id = child_id`, the self row. Every api has exactly one.
- `cpu_time` / `gpu_time` always describe the **child**, replicated to each ancestor.
- `gpu_time` = sum of durations of ops directly correlated to that child api.

Therefore:
- `SUM(gpu_time) GROUP BY parent_id` (all depths) = **inclusive** — all GPU work
  launched anywhere beneath that frame.
- `WHERE depth=0` = **exclusive / self** — only what the frame itself did.

```sql
-- inclusive GPU time per torch op; COUNT(DISTINCT parent_id) for real call counts
SELECT B.apiName, COUNT(DISTINCT A.parent_id) AS calls,
       SUM(A.gpu_time)/1e6 AS gpu_ms
FROM ext_callstack A JOIN api B ON B.id = A.parent_id
WHERE B.domain='torch' AND B.category='function'
GROUP BY 1 ORDER BY gpu_ms DESC LIMIT 5;
```
```
autograd::engine::evaluate_function: torch::autograd::AccumulateGrad|11978|6779.5
c10d::allreduce_|392|5437.9          record_param_comms|645|5225.5
aten::convolution_backward|4240|2696.6
autograd::engine::evaluate_function: MiopenBatchNormBackward0|6753|1123.6
```
(2.9 s — near the budget. Filter by `apiName` to keep it fast.)

**`COUNT(*)` here counts closure rows, not calls** — it would report 168,021
instead of 11,978. Always `COUNT(DISTINCT parent_id)`.

Exclusive self CPU time (finds *where the host is actually spending time*):
```sql
SELECT B.apiName, COUNT(*) AS calls, SUM(A.cpu_time)/1e6 AS self_cpu_ms
FROM ext_callstack A JOIN api B ON B.id = A.parent_id
WHERE A.depth=0 AND B.domain='torch'
GROUP BY 1 ORDER BY self_cpu_ms DESC LIMIT 5;
-- DistributedDataParallel.forward|80|3606.5    c10d::allgather_|2|2323.1
-- record_param_comms|649|1451.1
-- autograd::engine::evaluate_function: torch::autograd::AccumulateGrad|11978|1139.2
-- autograd::engine::evaluate_function: AddmmBackward0|93|825.6      (0.7 s)
```
Convenience views `callStack_inclusive`, `callStack_exclusive`,
`callStack_inclusive_name`, `callStack_exclusive_name` wrap these groupings.

## 11. Copy bandwidth

Two traps:

1. **The `copy` view contains non-copy APIs** — allocation and memset calls get a
   `rocpd_copyapi` row. Unfiltered totals are absurd:
   ```sql
   SELECT apiName, COUNT(*) n, SUM(size) bytes FROM copy GROUP BY 1 ORDER BY n DESC;
   -- hipMemcpyAsync|26614|8604723704     hipMemcpyWithStream|654|204903992
   -- hipMalloc|510|19564331008   <-- 19.5 GB that was never copied
   -- hipMemsetAsync|48|151110032         hipMemset|6|157286400
   ```
2. **`copy` is API timing; `copyop` is GPU timing.** For bandwidth use `copyop`.

Correct:
```sql
SELECT apiName, COUNT(*) AS copies, SUM(size)/1e9 AS total_GB,
       SUM(duration)/1e6 AS gpu_ms,
       SUM(size)*1.0/SUM(duration) AS GB_per_s   -- bytes/ns == GB/s
FROM copyop
WHERE apiName LIKE '%Memcpy%' AND duration > 0
GROUP BY 1 ORDER BY total_GB DESC;
-- hipMemcpyAsync|26612|8.60|114.1|75.4        hipMemcpyWithStream|636|0.09|14.6|6.3
```
Break down by direction with `kind, srcDevice, dstDevice`.

## 12. Kernel launches — do not filter on `hipLaunchKernel`

There are several launch entry points. Filtering on `hipLaunchKernel` alone
silently misses most launches (29,760 of 70,804 = 42%). **Join
`rocpd_kernelapi`, which is launch-API agnostic**, or use the `kernel` view.

```sql
SELECT C.string AS apiName, COUNT(*) n
FROM rocpd_kernelapi K
JOIN rocpd_api A ON A.id=K.api_ptr_id
JOIN rocpd_string C ON C.id=A.apiName_id
GROUP BY 1 ORDER BY n DESC;
-- hipExtModuleLaunchKernel|40480   hipLaunchKernel|29760   hipModuleLaunchKernel|564
```

**`gridX` semantics differ by launch path.** For `hipLaunchKernel` it is a
*block count*; for module-launch paths the source records work-items. Do not
compare `gridX` across launch APIs or compute occupancy from it without checking
which entry point was used.

## 13. Time-window scoping

To constrain analysis to a marked phase, materialize the ranges then join by
**temporal containment plus thread identity** (`rpd2table.py:54`). Markers are
thread-local; omitting `pid`/`tid` leaks unrelated work.

```sql
WITH m AS MATERIALIZED (
  SELECT id AS mid, pid, tid, start, end FROM api
  WHERE domain='torch' AND apiName='Optimizer.step#SGD.step'
)
SELECT O.description AS kernel, COUNT(*) AS launches, SUM(O.end-O.start)/1e6 AS gpu_ms
FROM m
JOIN rocpd_api a  ON a.pid=m.pid AND a.tid=m.tid
                 AND a.start>=m.start AND a.start<=m.end
JOIN rocpd_api_ops ao ON ao.api_id=a.id
JOIN op O ON O.id=ao.op_id
GROUP BY 1 ORDER BY gpu_ms DESC LIMIT 3;
-- multi_tensor_apply_kernel<...BinaryOpListAlphaFunctor<f,2,2,0>...>|474|75.0
-- multi_tensor_apply_kernel<...BinaryOpListAlphaFunctor<f,3,2,2>...>|320|38.8
-- multi_tensor_apply_kernel<...BinaryOpScalarFunctor<f,1,1,0>...>|234|22.6      (0.4 s)
```
`AS MATERIALIZED` matters — it stops SQLite re-evaluating the marker scan per row.
Aggregate with `COUNT(DISTINCT mid)` to count marker *instances* without fan-out
inflation.

Where `ext_callstack` exists it gives the same answer far more directly, and is
robust to nesting (137.85 ms for `Optimizer.step` by both methods):
```sql
SELECT COUNT(DISTINCT A.parent_id) AS instances, SUM(A.gpu_time)/1e6 AS gpu_ms
FROM ext_callstack A JOIN api B ON B.id=A.parent_id
WHERE B.domain='torch' AND B.apiName='Optimizer.step#SGD.step';
-- 80|137.85
```

For a coarse "utilization during phase X", bound the window and scan ops:
```sql
WITH w AS (SELECT MIN(start) t0, MAX(end) t1 FROM api
           WHERE domain='torch' AND apiName='DistributedDataParallel.forward')
SELECT O.gpuId, COUNT(*) AS ops, SUM(O.end-O.start)/1e6 AS gpu_ms,
       100.0*SUM(O.end-O.start)/(SELECT t1-t0 FROM w) AS busy_pct
FROM rocpd_op O, w
WHERE O.start BETWEEN w.t0 AND w.t1
  AND O.opType_id NOT IN (SELECT id FROM rocpd_string WHERE string='Barrier')
GROUP BY O.gpuId;
-- 2|40168|2814.2|67.7      3|40170|3014.6|72.5      (0.2 s)
```

## 14. Optional / generated tables

These exist **only if** the corresponding post-processing script was run.
Core `rocpd_*` tables and the base views are always present.

| Object | Produced by | Adds |
|---|---|---|
| `ext_callstack` (+ 4 `callStack_*` views) | `rocpd/call_stacks.py` | call hierarchy, exclusive/inclusive CPU & GPU time |
| `ext_graph`, `ext_graph_kernelapis`, views `graphLaunch`, `graphKernel` | `rocpd/graph.py` | HIP/CUDA graph capture and replay |
| `ext_autogradapi`, views `autograd`, `autogradKernel` | `rocpd/autograd.py` | autograd op → kernel attribution with tensor sizes |
| views `counter`, `counter_summary` | recent schema versions only | resolved HW counter values (§5) |

**Probe before querying** — one cheap query covers everything:
```sql
SELECT
 (SELECT COUNT(*) FROM sqlite_master WHERE name='ext_callstack')   AS has_callstack,
 (SELECT COUNT(*) FROM sqlite_master WHERE name='ext_graph')       AS has_graph,
 (SELECT COUNT(*) FROM sqlite_master WHERE name='ext_autogradapi') AS has_autograd,
 (SELECT COUNT(*) FROM sqlite_master WHERE name='counter')         AS has_counter_view,
 (SELECT COUNT(*) FROM rocpd_counter)    AS n_counters,
 (SELECT COUNT(*) FROM rocpd_monitor)    AS n_monitor,
 (SELECT COUNT(*) FROM rocpd_stackframe) AS n_stackframe;
-- reference trace: 1|0|0|0|0|0|0
```
`rocpd_counter`, `rocpd_monitor`, and `rocpd_stackframe` are core tables
(`tableSchema.cmd`) and always exist, so counting them is always safe — it is the
`counter` *views* that may be missing.
`rocpd_metadata` also records generation: tags `Callstack::Generated`,
`Graph::Generated`, `Autograd::Generated`.

## 15. Performance notes

Most `.rpd` files ship with **no analysis indexes** (the tracer omits them to
keep capture fast) — the reference trace has only two autoindexes. So:

- Aggregate in SQL; never dump raw rows and post-process.
- `json_extract` over a whole table costs ~0.5–1 s per 700 k rows. Filter by
  `domain` first.
- Use `WITH ... AS MATERIALIZED` for any CTE referenced more than once.
- Window functions and delta-scans are linear; **correlated subqueries over
  `rocpd_op` are quadratic and will time out**.
- Joining all of `ext_callstack` to `api` is ~1.8 s for 2.6 M rows — acceptable,
  but add a filter when you can.

Orientation query, run it first (0.2 s):
```sql
SELECT (SELECT COUNT(*) FROM rocpd_api) AS apis,
       (SELECT COUNT(*) FROM rocpd_op)  AS ops,
       (SELECT COUNT(DISTINCT pid) FROM rocpd_api) AS procs,
       (SELECT COUNT(DISTINCT gpuId) FROM rocpd_op) AS gpus,
       (SELECT (MAX(end)-MIN(start))/1e9 FROM rocpd_api) AS wall_s,
       (SELECT SUM(end-start)/1e9 FROM rocpd_op) AS gpu_s;
-- 710227|99146|3|2|11.40|12.26
```

---

# PART II — APPLICATION DOMAINS OBSERVED IN THE REFERENCE TRACE

**Observed, not guaranteed.** The reference trace is a 2-GPU distributed
ResNet-style PyTorch training run. What follows is legitimate evidence for how
PyTorch and MIOpen *typically* log, and you should use it confidently once a
discovery query confirms the domain is present. It does **not** constrain what
other applications may log.

Start every session with the §6 domain inventory. Reference trace:
`torch` (function / backward_function / user_scope), `miopen` (function /
driver), `hip`, `rpd_tracer`. No `roctx`/`nvtx`, no counters, no monitor data.

### `args` format by domain — check this before using `json_extract`
```sql
SELECT domain, category,
       CASE WHEN args='' THEN 'empty' WHEN json_valid(args) THEN 'json' ELSE 'plain' END AS fmt,
       COUNT(*) n
FROM api GROUP BY 1,2,3 ORDER BY n DESC;
```
```
torch|function|json|334290        miopen|function|json|211792
hip||json|71314                   hip||empty|43263
torch|backward_function|json|27120   miopen|driver|plain|21256
torch|user_scope|json|884         rpd_tracer|overhead|plain|308
```

## II.1 — `torch`

Discovery:
```sql
SELECT category, COUNT(*) FROM api WHERE domain='torch' GROUP BY 1;
-- function|334290   backward_function|27120   user_scope|884
```

Three categories:
- **`function`** — forward-pass aten ops and autograd engine frames. `apiName` is
  the op (`aten::conv2d`, `autograd::engine::evaluate_function: ...`).
- **`backward_function`** — autograd backward nodes (`ConvolutionBackward0`).
- **`user_scope`** — higher-level phase markers: `DistributedDataParallel.forward`,
  `Optimizer.step#SGD.step`, `nccl:all_reduce`, `nccl:broadcast`.

`args` is JSON with a stable shape:
```json
{"seq":177,"op_id":7787,"sizes":[[]]}
{"seq":-1,"op_id":3,"sizes":[[],[],[],[],[],[]]}
```

| Key | Meaning |
|---|---|
| `seq` | autograd sequence number; **`-1` means not in a sequence** |
| `op_id` | framework-internal operator id — **NOT a `rocpd_op.id`** |
| `sizes` | input tensor shapes, list of lists |

**`$.op_id` does not join to `op.id`.** Verified: the join returns **0 rows**, and
the ranges are disjoint (`op_id` 1…181,147 vs `rocpd_op.id` 4.29e9…8.59e9). It is
a framework-internal counter. To link a torch annotation to GPU work, use
`ext_callstack` (§10) or temporal containment (§13).

### Tensor shapes per op
Instance of the Part I *`json_extract` on `args`* mechanism (§8):
```sql
SELECT apiName, json_extract(args,'$.sizes') AS sizes, COUNT(*) n,
       SUM(end-start)/1e6 AS cpu_ms
FROM api WHERE domain='torch' AND apiName='aten::conv2d'
GROUP BY 1,2 ORDER BY n DESC LIMIT 3;
-- aten::conv2d|[[32,256,14,14],[1024,256,1,1],...]|480|47.8
-- aten::conv2d|[[32,1024,14,14],[256,1024,1,1],...]|400|57.4
-- aten::conv2d|[[32,256,14,14],[256,256,3,3],...]|400|39.3      (0.2 s)
```

### Forward/backward pairing on `$.seq`
Also an instance of §8: torch pairs a forward op with its backward node via a
shared `seq` key. **Filter `seq >= 0`** — 295,790 `function` and 12,880
`backward_function` rows carry the `-1` sentinel and would otherwise all collapse
onto one key.

`seq` is **not unique**: 38,500 forward rows share only 7,120 distinct values.
`tracing.py` resolves this by keeping the **last** forward op per `seq`:

```sql
WITH t AS MATERIALIZED (
  SELECT id, category, apiName, json_extract(args,'$.seq') AS seq
  FROM api WHERE domain='torch' AND category IN ('function','backward_function')
    AND json_valid(args) AND json_extract(args,'$.seq') >= 0),
fwd AS (SELECT seq, apiName,
        ROW_NUMBER() OVER (PARTITION BY seq ORDER BY id DESC) rn
        FROM t WHERE category='function')
SELECT f.apiName AS fwd, b.apiName AS bwd, COUNT(*) AS pairs
FROM t b JOIN fwd f ON f.seq=b.seq AND f.rn=1
WHERE b.category='backward_function'
GROUP BY 1,2 ORDER BY pairs DESC LIMIT 5;
-- aten::conv2d|ConvolutionBackward0|4234
-- aten::relu_|AddmmBackward0|3920
-- aten::batch_norm|MiopenBatchNormBackward0|3258
-- aten::add_|AddmmBackward0|1280
-- aten::batch_norm|torch::autograd::AccumulateGrad|959      (0.6 s)
```
A plain `JOIN ... ON b.seq=f.seq` without the `rn=1` restriction produces a
cross-product per `seq` and **did not finish in 120 s**. Always dedupe first.

### Phase markers (`user_scope`)
```sql
SELECT apiName, COUNT(*) n, SUM(end-start)/1e6 AS ms
FROM api WHERE domain='torch' AND category='user_scope'
GROUP BY 1 ORDER BY ms DESC;
-- DistributedDataParallel.forward|80|3606.5    Optimizer.step#SGD.step|80|344.7
-- Optimizer.zero_grad#SGD.zero_grad|160|28.8   nccl:all_reduce|392|25.2
-- nccl:broadcast|166|11.2                      nccl:all_gather|6|17.7
```
These are the natural windows for §13 scoping. 80 instances = 80 training steps.
`Optimizer.zero_grad` launches no GPU work (scoping it returns 0 ops) — a real
result, not a bug.

## II.2 — `miopen`

Discovery:
```sql
SELECT category, COUNT(*) FROM api WHERE domain='miopen' GROUP BY 1;
-- function|211792   driver|21256
```

- **`function`** — the MIOpen C API. `args` is JSON, but values are **strings**,
  including pre-formatted tensor descriptors:
  ```json
  {"handle":"stream: 0, device_id: 0",
   "xDesc":"{32, 3, 224, 224}, {150528, 50176, 224, 1}, packed, ",
   "wDesc":"{64, 3, 7, 7}, {147, 49, 7, 1}, packed, ",
   "convDesc":"conv2d, miopenConvolution, miopenPaddingDefault, {3, 3}, {2, 2}, {1, 1}, ",
   "algo":"1", "workSpaceSize":"0"}
  ```
  Extract with `json_extract(args,'$.xDesc')`, then parse the shape as text.
  Highest-volume calls are descriptor create/destroy churn
  (`miopenCreateTensorDescriptor` 54,880) — noise, not work.

- **`driver`** — `args` is a **plain `MIOpenDriver` command line, not JSON**.
  These are zero-duration marks that reproduce the exact convolution/BN problem.
  ```sql
  SELECT args, COUNT(*) n FROM api
  WHERE domain='miopen' AND apiName='LogCmdConvolution'
  GROUP BY 1 ORDER BY n DESC LIMIT 2;
  -- ./bin/MIOpenDriver conv -n 32 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 ... -F 4 -t 1|480
  -- ./bin/MIOpenDriver conv -n 32 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 ... -F 2 -t 1|480
  ```
  `-F` is the direction (1 fwd, 2 bwd-data, 4 bwd-weights). This is the best
  source for "which conv shapes does this model run" and is directly runnable to
  reproduce a kernel in isolation. `apiName` values: `LogCmdConvolution` (12,640),
  `LogCmdBNorm` (8,480), `LogCmdFindConvolution` (136).

## II.3 — `hip`

Empty-string category. 71,314 rows have JSON `args`, 43,263 have `''`.
Where present, `args` is mostly an opaque packed-argument blob
(`{"args":"0x0800..."}`) — **not useful for analysis**. Use `rocpd_kernelapi` /
`rocpd_copyapi` for launch and copy parameters instead (§12, §11).

## II.4 — `rpd_tracer` — profiler overhead, not application activity

```sql
SELECT apiName, COUNT(*) n, SUM(end-start)/1e6 AS ms
FROM api WHERE domain='rpd_tracer' GROUP BY 1 ORDER BY ms DESC;
-- ApiTable::writeRows|171|1682.1        ClrDataSource::chunkCallback|90|763.4
-- OpTable::writeRows|22|314.1           KernelApiTable::writeRows|16|164.1
-- CopyApiTable::writeRows|6|62.4        ClrDataSource::flush|3|32.9
```
Total 3,019 ms = **26.5% of wall time**. Report this as *measurement overhead*
when explaining where wall time went; never as application work. `args` is
`k=v | k=v` plain text.

## II.5 — Distributed / multi-process shape

Two ranks, one GPU each, correlated by `pid` and by `id>>32` (§3):
```sql
SELECT A.pid, O.gpuId, COUNT(*) ops, SUM(O.end-O.start)/1e6 AS gpu_ms
FROM rocpd_api_ops X JOIN rocpd_api A ON A.id=X.api_id JOIN rocpd_op O ON O.id=X.op_id
GROUP BY 1,2;
-- 331100|2|49566|5972.2      331101|3|49580|6290.9
```
Communication kernels land on a **dedicated queue** (`queueId=4`) — visible in
the per-queue breakdown and useful for separating compute from comms:
```sql
SELECT gpuId, queueId, COUNT(*) ops, SUM(end-start)/1e6 AS gpu_ms
FROM rocpd_op GROUP BY 1,2 ORDER BY 1,2;
-- gpu2: q0 48953/3250.8   q4 565/2721.1  (nccl)   q1-q3 negligible
-- gpu3: q0 48967/3365.9   q4 565/2924.7
```
For genuinely merged multi-node traces use the `napi` / `nop` views, which decode
`node` from `pid`/`gpuId` using the stride metadata (§5).

---

# WORKED EXAMPLES

### Q: "How much GPU time did `aten::mm` account for?"
Needs `ext_callstack` (§10) — a torch annotation launches no ops itself, so
`rocpd_api_ops` alone gives nothing.
```sql
SELECT B.apiName, COUNT(DISTINCT A.parent_id) AS calls,
       SUM(A.gpu_time)/1e6 AS gpu_ms,
       SUM(CASE WHEN A.depth=0 THEN A.cpu_time ELSE 0 END)/1e6 AS self_cpu_ms
FROM ext_callstack A JOIN api B ON B.id=A.parent_id
WHERE B.domain='torch' AND B.apiName IN ('aten::mm','aten::addmm','aten::conv2d')
GROUP BY 1 ORDER BY gpu_ms DESC;
-- aten::conv2d|4240|1020.65|15.95
-- aten::mm|160|3.74|32.15
-- aten::addmm|80|3.52|580.49      (1.8 s)
```
`aten::mm` is negligible here (3.7 ms); convolution dominates. Note `aten::addmm`
spends 580 ms of *host* time for 3.5 ms of GPU work — CPU-bound.

### Q: "Which kernels dominate, and what fraction of GPU time?"
```sql
WITH k AS (SELECT description AS kernel, COUNT(*) calls, SUM(end-start) ns
           FROM op WHERE opType='KernelExecution' GROUP BY 1)
SELECT substr(kernel,1,45) AS kernel, calls, ns/1e6 AS gpu_ms,
       ROUND(100.0*ns/SUM(ns) OVER (),2) AS pct,
       ROUND(100.0*SUM(ns) OVER (ORDER BY ns DESC)/SUM(ns) OVER (),2) AS cum_pct
FROM k ORDER BY ns DESC LIMIT 6;
```
```
_Z23ncclDevKernel_Generic_224ncclDevKernelArg|564 |890.50|12.90|12.90
MIOpenBatchNormBwdSpatial                    |4240|594.18| 8.61|21.51
miopenGcnAsmConv1x1WrW                       |560 |417.38| 6.05|27.55
miopenSp3AsmConv3x3F                         |1200|406.19| 5.88|33.44
_ZN2at6native29vectorized_elementwise_kernelI|2560|389.52| 5.64|39.08
_ZN2at6native29vectorized_elementwise_kernelI|3920|364.92| 5.29|44.36
```
(0.1 s.) Flat profile — top 6 are only 44%. Restricting to `KernelExecution`
excludes the `Barrier` distortion (§7).

Rows 5 and 6 are **different kernels** that `substr()` truncated to the same
prefix. Always `GROUP BY` the full `description` and truncate only for display —
never group on a truncated name.

### Q: "What was GPU utilization during the DDP forward phase?"
See §13 — 67.7% / 72.5% on the two GPUs over a 4.16 s window.

### Q: "Are short kernels a problem?"
```sql
SELECT COUNT(*) AS n, SUM(end-start)/1e6 AS ms,
  100.0*COUNT(*)/(SELECT COUNT(*) FROM rocpd_op
    WHERE opType_id NOT IN (SELECT id FROM rocpd_string WHERE string='Barrier')) AS pct_calls,
  100.0*SUM(end-start)/(SELECT SUM(end-start) FROM rocpd_op
    WHERE opType_id NOT IN (SELECT id FROM rocpd_string WHERE string='Barrier')) AS pct_time
FROM rocpd_op
WHERE (end-start) < 10000
  AND opType_id NOT IN (SELECT id FROM rocpd_string WHERE string='Barrier');
-- 49599|205.1|50.55|2.92      (0.1 s)
```
Half of all GPU ops are under 10 µs but account for only 2.9% of GPU time — a
launch-overhead and fusion opportunity, not a duration problem.

### Q: "Where did wall time go?"
```sql
SELECT domain, category, COUNT(*) AS calls, SUM(end-start)/1e6 AS cpu_ms
FROM api GROUP BY 1,2 ORDER BY cpu_ms DESC;
-- torch|function|334290|22672.8      hip||114577|5826.2
-- torch|user_scope|884|4034.2        rpd_tracer|overhead|308|3019.0
-- torch|backward_function|27120|2515.1   miopen|function|211792|1407.3
-- miopen|driver|21256|0.0
```
(1.0 s.) **These sum to far more than the 11.4 s wall span because CPU ranges
nest** (§7) — read them as per-layer inclusive totals, not a partition. For a
true breakdown use exclusive time (§10). And note 3.0 s of that is `rpd_tracer`
measuring itself.

---

All row counts and values above come from one reference trace and are
illustrative. Every query shown was executed against it and completes well within
the 10 s budget; each is written to regenerate the numbers for whatever trace is
loaded.
