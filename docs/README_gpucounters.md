# GPU Hardware Counter Collection

rpd_tracer can collect GPU hardware performance counters (e.g. ALU utilization, cache hit rates, wave counts) alongside the normal trace data. Counter values are recorded per kernel dispatch and stored in the rpd database.

This feature requires the `RocprofDataSource` backend (rocprofiler-sdk). It may not be selected by default.

## Enabling Counter Collection

Two settings are needed: select the RocprofDataSource backend and turn on counter collection.

### Using environment variables

```bash
RPDT_DATASOURCES_PRIORITY=RocprofDataSource \
RPDT_ROCPROF_COLLECT_COUNTERS=1 \
runTracer.sh python my_script.py
```

### Using rlog-config (persistent)

```bash
rlog-config set rpd_tracer datasources_priority RocprofDataSource
rlog-config set rpd_tracer rocprof_collect_counters 1
```

Then run normally:

```bash
runTracer.sh python my_script.py
```

To disable counter collection later:

```bash
rlog-config set rpd_tracer rocprof_collect_counters 0
```

## Querying Counter Data

Counter values are stored in the `rocpd_counter` table, which links each value to a kernel dispatch via `op_id` and to a counter name via `name_id`.

Two convenience views are created automatically:

**`counter`** — one row per counter per dispatch, joined to kernel name:

```sql
select * from counter where kernelName like '%gemm%' limit 10;
```

| op_id | kernelName | counterName | value |
|-------|------------|-------------|-------|
| 42 | Cijk_... | VALUBusy | 87.3 |
| 42 | Cijk_... | VALUInsts | 12045.0 |

**`counter_summary`** — aggregated by kernel name, showing avg/min/max across all dispatches:

```sql
select * from counter_summary;
```

| kernelName | counterName | dispatches | avg | min | max |
|------------|-------------|------------|-----|-----|-----|
| Cijk_... | VALUBusy | 500 | 87.3 | 72.1 | 94.5 |

## Customizing Which Counters Are Collected

By default, two counter sets are collected:

- Set 1: `VALUInsts`, `VALUBusy`, `VALUUtilization`
- Set 2: `SQ_WAVES`, `SALUInsts`, `MemUnitBusy`

### Discovering available counters

The available counters depend on your GPU. To list them:

```bash
rocprofv3 -L
```

This prints every counter name with a description. There are typically hundreds. The derived counters (e.g. `VALUBusy`, `L2CacheHit`, `MemUnitBusy`) are generally the most useful — they are human-readable metrics computed from raw hardware registers.

### Overriding the default counter sets

Set the `rocprof_counter_sets` property to a custom value. Commas separate counters within a set. Semicolons separate sets.

```bash
# Single set of two counters
RPDT_ROCPROF_COUNTER_SETS="L2CacheHit,MemUnitBusy"

# Two sets, alternating between dispatches
RPDT_ROCPROF_COUNTER_SETS="VALUBusy,SALUBusy;L2CacheHit,FetchSize,WriteSize"
```

Or persistently:

```bash
rlog-config set rpd_tracer rocprof_counter_sets "VALUBusy,SALUBusy;L2CacheHit,FetchSize,WriteSize"
```

To revert to defaults, clear the property:

```bash
rlog-config set rpd_tracer rocprof_counter_sets ""
```

### Hardware limitations on counter combinations

GPU performance counters are backed by a limited number of hardware registers. Not all counters can be collected simultaneously — counters that share the same hardware block may conflict. If you request a set of counters that exceeds the hardware limit, the set will fail to create and no counters will be collected for those dispatches. A warning is logged when this happens.

There is no way to query which combinations are valid ahead of time. The general rules:

- Counters from different hardware blocks (e.g. one SQ counter + one TCC counter) are more likely to work together.
- Counters from the same block (e.g. multiple SQ counters) may conflict if they exceed the block's register budget.
- Derived counters (like `VALUBusy`) may internally depend on multiple raw counters, consuming more registers than expected.
- Fewer counters per set is safer.

If you need many counters, split them across multiple sets rather than putting them all in one.

### How multiple sets work (round-robin)

When multiple counter sets are configured, each kernel dispatch collects only one set at a time. The sets rotate per kernel: the first dispatch of a given kernel uses set 1, the second uses set 2, then back to set 1, and so on.

This means that over many iterations (e.g. a training loop), every kernel eventually gets measured with every counter set. The `counter_summary` view averages across all dispatches that collected a given counter, giving a representative picture even though individual dispatches only collected a subset.

This approach avoids the hardware conflict problem entirely — each set can be kept small enough to fit within hardware limits, while the full collection of sets covers all the counters you care about.
