# RPD_INFO.md Research Prompt

This is the prompt used to generate `rpd_dash/RPD_INFO.md`, the schema reference
injected into the chat assistant's system prompt. Re-run it against a reference
`.rpd` trace and the full source tree to regenerate that file.

The prompt is deliberately trace-agnostic — it never names a specific `.rpd`.
Re-running it with a richer trace (more application domains, graph/autograd
tables generated, monitor/counter data present) should expand Part II without
requiring any edits to the prompt itself.

---

# Task

Produce `RPD_INFO.md`: a schema and analysis reference for RPD (ROCm Profile Data)
SQLite trace files. It is injected verbatim into the system prompt of a chat
assistant that answers user questions about a trace by writing SQL against it.

You are given a reference `.rpd` trace and the full source tree. Write the file
that makes that assistant fast, correct, and confident.

## Consumer constraints (these are hard requirements)

The assistant runs each query through a read-only SQLite connection with:
- `QUERY_TIMEOUT_S = 10` — **every query you write must complete in under 10 seconds**
- `ROW_LIMIT = 500`, results truncated at 6000 chars — prefer aggregates over raw dumps
- `PRAGMA query_only = ON` — read-only; never suggest writes, `CREATE`, or `ATTACH`

The file is prepended to every chat request, so tokens cost per-message. Be dense
and useful; omit filler. Target roughly 400–600 lines.

---

# Part I vs Part II: the organizing principle

Structure the document in two clearly labeled parts. Most prior attempts at this
file failed by conflating them.

**Part I — Invariant schema.** True of *any* `.rpd`, regardless of what was
profiled: units, string interning, table relationships, join paths, the built-in
views, the domain taxonomy, optionally-generated tables. State these
unconditionally.

**Part II — Application domains observed in the reference trace.** Contingent on
what the profiled application logged: which domains are present, their `args`
JSON shapes, app-specific correlation patterns.

Rules binding the two:

1. **Every Part II pattern must name the Part I mechanism it rides on**, so a
   reader on a different application can transfer it. (e.g. "torch pairs forward
   and backward ops on a shared `$.seq` key — an instance of the general
   *correlate annotations via `json_extract` on `args`* mechanism.")
2. **Open each Part II subsection with the discovery query** that establishes
   whether it applies to the loaded trace.
3. **The example is just the example.** The reference trace shows what PyTorch
   and MIOpen *typically* log. That is legitimate evidence for answering
   questions about those domains — use it, and be concrete. It is **not** the
   final word on the schema, and it does not constrain what other applications
   may log. Label Part II as observed-not-guaranteed, then let the assistant rely
   on it once a discovery query confirms presence. Do not hedge so heavily that
   the assistant becomes evasive about the common case.

---

# Stated facts: the domain taxonomy (Part I)

Do not spend research effort rediscovering this. Verify and explain it.

`rocpd_api` rows carry `(domain, category, apiName, args)`. Domains fall into
four kinds:

| Kind | Domains | Meaning |
|---|---|---|
| **Driver** | `hip`, `cuda` | CPU-side calls that dispatch and manage GPU work. The bridge from host to device. Not application semantics. |
| **Marker / annotation** | `roctx`, `nvtx` | User-inserted ranges and marks. Conventionally `apiName = 'UserMarker'`, with `category` of `range` or `mark`. |
| **Higher-order (rlog)** | caller-defined — e.g. `torch`, `miopen` | Library/framework instrumentation via the rlog API. |
| **Profiler self-instrumentation** | `rpd_tracer` | The tracer measuring its own overhead (`category = 'overhead'`). Not application activity — say so, so it is never misreported as such. |

Key Part I consequence to make explicit: in `RlogDataSource.cpp` (see `mark()`
and `rangePush()`), `domain`, `category`, and `apiName` are **arbitrary
caller-supplied strings** interned at call time. This is how `torch` and `miopen`
appear without any roctx/nvtx involvement. **The set of domains is therefore
open-ended and unpredictable.**

Because of that, the assistant's **standing first move on any unfamiliar trace**
is to enumerate what is actually present. Put this query early and prominently:

```sql
SELECT domain, category, COUNT(*) AS calls
FROM api GROUP BY domain, category ORDER BY calls DESC;
```

Also note: the `hip` domain uses an **empty-string** `category` (`''`), not NULL.
Any category filter or `GROUP BY` must account for it.

---

# Research: analysis techniques to mine

The assistant must answer common, medium-difficulty questions **by reflex**,
without deriving them from first principles. Mine these sources for canonical,
working patterns and include them as ready-to-run SQL:

**Schema and views**
- `rocpd_python/rocpd/schema_data/*.cmd` — authoritative table and view DDL.
  Read every view definition; they encode the canonical join paths.

**Analysis scripts**
- `rocpd_python/rocpd/call_stacks.py` — call-hierarchy materialization; exclusive
  vs inclusive CPU/GPU time; the `ext_callstack` parent/child model.
- `rocpd_python/rocpd/tracing.py` — **queue-depth and GPU-idle fabrication** via
  a linear delta-scan (+1 at op start, −1 at op end, ordered by timestamp,
  running sum), and **forward/backward flow pairing** by shared `seq`. Note the
  schema-version branch for how markers are located.
- `rocpd_python/rocpd/graph.py`, `autograd.py`, `subclass.py` — derived tables
  and the views they add.
- `tools/rpd2table.py`, `tools/rpd_marker_summary.py` — **time-window scoping**:
  locating markers and constraining analysis to their `start`/`end` range.
- `tools/rpd_trim.py`, `helpful_queries/*.cmd` — additional idioms.

**Queries the product already ships** (high value — these are the questions users
actually ask):
- `rpd_dash/rpd_dash/pages/*.py` and `rpd_dash/rpd_dash/util/fragments.py`
  contain the SQL behind kernel ranking, API/op summaries, copy analysis,
  short-kernel analysis, torch-op breakdowns, counters, and busy stats.

Cover at minimum: time-window scoping, call-hierarchy attribution, queue depth
and utilization, kernel and op ranking, copy bandwidth, forward/backward pairing,
and per-domain annotation analysis.

---

# Optional / generated tables

`ext_callstack`, `ext_graph` (+ `graphLaunch`, `graphKernel`), `ext_autogradapi`
(+ `autograd`, `autogradKernel`) exist **only if** the corresponding
post-processing script has been run on the trace. The core `rocpd_*` tables and
base views are always present.

Give these their own Part I section with an existence guard the assistant can run
before querying, e.g.:

```sql
SELECT name FROM sqlite_master WHERE name = 'ext_callstack';
```

---

# Verification protocol (this is where prior attempts failed)

1. **Execute every SQL example against the reference trace.** Paste real
   representative output as evidence. **Do not include any query you have not
   run.**
2. **Time every query. Reject anything over 10 seconds.** Correlated-subquery
   self-joins over the op table do not scale — use window functions or
   delta-scans instead.
3. **A pattern returning 0 rows is not necessarily invalid.** Before removing or
   contradicting one, check the emitting source (`rpd_tracer/*DataSource.cpp`,
   `rocpd_python/rocpd/*.py`). Distinguish *wrong* from *not exercised by this
   workload*, and label which. Absence of evidence in one trace is not evidence
   of absence in the schema.
4. **Verify claimed joins actually return rows** before asserting them.
5. Where a column is a sentinel or has surprising semantics, say so explicitly.

## Known defects in the current file — verified; do not reproduce

Each was confirmed by execution against the reference trace:

- **`json_extract(args,'$.op_id')` does NOT join to `op.id`.** That join returns
  zero rows; the value ranges are disjoint. It is a framework-internal operator
  id, not a `rocpd_op` key. Explain the correct way to link an annotation to GPU
  work (temporal containment, or `ext_callstack` where available).
- **`busy.Busy` is not a 0–1 fraction.** Observed values exceed 1.0, because
  `WallTime` is global across all GPUs while `GpuTime` sums overlapping ops.
  Document the real semantics and give a correct per-GPU formulation.
- **The queue-depth self-join does not finish** (killed at 300s). Replace with
  the linear delta-scan from `tracing.py`.
- **Filtering kernel launches on `hipLaunchKernel` alone silently misses the
  majority of launches** — other launch entry points exist. Prefer joining
  `rocpd_kernelapi`, which is launch-API agnostic.
- **The `copy` view contains non-copy APIs** (allocation and memset calls).
  Computing bandwidth without filtering produces absurd results. Note also that
  `copy` reflects *API* timing while `copyop` reflects *GPU* timing.
- **`$.seq` uses `-1` as a sentinel.** Any pairing or grouping must filter
  `>= 0` or silently corrupt results.

---

# Deliverables

Write `RPD_INFO.md` containing:

1. **Part I — Invariant schema**
   - Units and time handling
   - Tables, their relationships, and the string-interning model
   - The `rocpd_api` → `rocpd_api_ops` → `rocpd_op` bridge (CPU dispatch → GPU
     execution) and how to traverse it in both directions
   - Built-in views with their real definitions and correct semantics
   - The four-kind domain taxonomy, with the domain-inventory query
   - `args` as JSON and the `json_extract` idiom
   - Optional/generated tables with existence guards
   - Canonical analysis patterns (time-window scoping, call hierarchy, queue
     depth/utilization, ranking, bandwidth), each with tested SQL

2. **Part II — Application domains observed in the reference trace**
   - Domain inventory first
   - Per-domain `args` shapes and what each library logs
   - Forward/backward pairing and other correlation patterns, each citing its
     Part I mechanism
   - Clearly labeled as observed, not guaranteed

3. **Worked examples** — 3–5 realistic user questions with tested SQL and real
   output, of the kind users actually ask. For example:
   - "How much GPU time did `aten::mm` account for?"
   - "Which kernels dominate, and what fraction of total GPU time?"
   - "What was GPU utilization during a specific marked phase?"

Row counts and value samples from the reference trace are illustrative. Present
them as such, and include the query that regenerates them for the loaded trace.
