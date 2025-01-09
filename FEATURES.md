# New Features

--------------------------------------------------------------------------------

This is a changelog that describes new features as they are added.  Newest first.


Contents:
<!-- toc -->

- [Remote Start/stop](#remote-startstop)
- [Graph Subclass](#graph-subclass)
- [Pytorch Autograd Subclass](#pytorch-autograd-subclass)
- [Call Stack Accounting](#call-stack-accounting)
- [Rpd_tracer Start/stop](#rpd_tracer-startstop)
- [Schema v2](#schema-v2)

<!-- tocstop -->


--------------------------------------------------------------------------------
## Remote Start/stop
Recording can be started and stoped externally through a loader, librpd_remote.so

The loader library is low overhead and can be linked against directly or used with LD_PRELOAD.
```
LD_PRELOAD=librpd_remote.so python3
```
To record, first create an empty rpd file in the processes' working directory.  The tracer will only append trace data to an existing file, it will not overwrite.
```
python rocpd.schema --create trace.rpd
```
If in doubt, you can discover this directory before hand by doing a dry run and looking for the zero-length trace file created.  That is the location for the initialized rpd file.

Once the process is running you can use the rpdRemote command to start/stop recording.
```
rpdRemote start <pid>
rpdRemote stop <pid>
```
Data is flushed each time the trace is stopped and can be inspected.

Limitations:
- Once the tracer is loaded, on the first 'start', it can not be unloaded.  There will be a slight increased in overhead.
- Once loaded, the tracer will record into the same file for the duration.  You can not replace the file on the fly.


## Graph Subclass
Additional graph analysis is available via a post-processing tool.  This uses graph api calls to identify graph captures, kernels, and subsequent launches.  

#### Augment an rpd file with graph info
```
python -m rocpd.graph <inputfile.rpd>
```

This creates and "ext_graph" table with an entry for each captured graph.  It also creates a view to show the kernel makeup of each graph and a view to show timing data for each launch.

#### graphLaunch view
This view contains information for each graph launch that occured.  This includes wall-time spent launching the graph, wall-time spent executing the kernels on the gpu, and the total gpuTime (busy time).

```
sqlite> select * from graphLaunch;
pid  tid  apiName         graphExec       stream  gpuId  queueId  apiDuration  opDuration  gpuTime
---  ---  --------------  --------------  ------  -----  -------  -----------  ----------  -------
713  713  hipGraphLaunch  0x5602c48aa700  0x0     2      0        2808         3689        246
713  713  hipGraphLaunch  0x5602c48aa700  0x0     2      0        245          471         166
713  713  hipGraphLaunch  0x5602c48aa700  0x0     2      0        200          453         152
713  713  hipGraphLaunch  0x5602c48aa700  0x0     2      0        212          452         153
713  713  hipGraphLaunch  0x5602c48aa700  0x0     2      0        199          452         152
713  713  hipGraphLaunch  0x5602c48aa700  0x0     2      0        195          466         158

```

#### graphKernel view
This view shows the kernel composition of each graph, including launch parameters.

```
sqlite> select * from graphkernel;

graphExec         sequence  kernelName                      gridX  gridY  gridZ  workgroupX  workgroupY ...
----------------  --------  ------------------------------  -----  -----  -----  ----------  ----------
0x5602c5107450    1         void at::native::legacy::eleme  1      1      1      128         1
                            ntwise_kernel<128, 4, at::nati

0x5602c5107450    2         Cijk_Alik_Bljk_SB_MT32x32x16_S  128    1      1      128         1
                            E_1LDSB0_APM1_ABV0_ACED0_AF0EM

0x5602c5107450    3         void at::native::(anonymous na  1      1      1      256         1
                            mespace)::fused_dropout_kernel

0x5602c5107450    4         void at::native::legacy::eleme  1      1      1      128         1
                            ntwise_kernel<128, 4, at::nati

0x5602c5107450    5         Cijk_Alik_Bljk_SB_MT32x32x16_S  128    1      1      128         1
                            E_1LDSB0_APM1_ABV0_ACED0_AF0EM

```



--------------------------------------------------------------------------------
## Pytorch Autograd Subclass
Autograd analysis is available via a post-processing tool.  This is designed to work with the torch.autograd.profiler.emit_nvtx() feature.  The UserMarker ranges inserted by the pytorch profiler are formatted and presented in a dedicated 'api-subclass' table.  This allows for querying against thing like operator name, kernel name, or tensor size.  This makes heavy use of call stack accounting to compute execution times based on operator, kernel, and tensor size.

#### Augment an rpd file with autograd info
```
python -m rocpd.autograd <inputfile.rpd>
```

This creates an "ext_autogradapi" subclass table that contains (autogradName, seq, op_id, sizes, input_op_ids) columns for each autograd api call.  It also creates some intermediate views with per event callstack data + autograd info.  For advanced users.  

#### autogradKernel view
This view contains an aggregated summary of cpu + gpu usage based on the (autograd function, kernel name, tensor size) triple.  This lets you see which kernels each operator is using and what tensor sizes are in play.

```
sqlite> select * from autogradKernel;
autogradName     kernelName                      sizes                      calls       avg_gpu     total_gpu
---------------  ------------------------------  -------------------------  -------  ----------  ----------
aten::addmm      void at::native::legacy::elem   [[768], [12, 768], [768,   5        3328.0      16640
aten::add_       void at::native::modern::elem   [[30522, 768], [], []]     5        180640.8    903204
aten::mul        void at::native::modern::elem   [[12, 1, 1, 224], []]      5        2048.0      10240
aten::addcmul_   void at::native::modern::elem   [[512, 768], [512, 768],   5        1536.0      7680
aten::fill_      void at::native::modern::elem   [[512, 768], []]           10       1392.0      13920
aten::addmm      void at::native::legacy::elem   [[3072], [2688, 768], [76  60       34909.3333  2094560
aten::mm         Cijk_Ailk_Bljk_SB_MT128x96x16   [[2688, 768], [768, 768]]  240      118810.116  28514428
aten::addcdiv_   void at::native::modern::elem   [[512, 768], [512, 768],   5        4000.0      20000
aten::index_sel  void at::native::(anonymous n   [[30522, 768], [], [2688]  5        34336.0     171680
aten::bmm        Cijk_Ailk_Bljk_SB_MT64x32x16_   [[144, 224, 224], [144, 2  60       60690.7333  3641444
```

#### Html summary
There is a (rough) sample tool to format the autogradKernel table as interactive html:
```
python tools/rpd_autograd_summary.py trace.rpd autograd.html
```


--------------------------------------------------------------------------------
## Call Stack Accounting
Callstack analysis is available via a post-processing tool.  This materializes the caller/callee relationship of cpu events. It also attaches the cpu and gpu execution time of each event.  This allow computing inclusive and exclusive times for each function.  Instrumented applications can use roctx/nvtx ranges to pass function call information into the profile.

#### Augment an rpd file with callstack info

```
python -m rocpd.callstack <inputfile.rpd>
```

This creates an "ext_callstack" table (which you will probably not use directly).  It contains the cpu + gpu time for each function and mirrors that time up to each parent caller.

#### callStack_inclusive_name view
This view contains each cpu function with total time spent by it and it's children

#### callStack_exclusive_name view 
This view contains only the 'exclusive' time spent by each function.  This does not count time used by any child functions it called

```
sqlite> select * from callStack_inclusive_name order by random()
parent_id     apiName                    args                       cpu_time    gpu_time
------------  -------------------------  -------------------------  ----------  ----------
7969          hipLaunchKernel                                       3620        1280
2147505467    UserMarker                 aten::add_                 11080       6240
2147508236    UserMarker                 aten::addcdiv_             24170       1280
2147508777    UserMarker                 aten::zero_                20750       1760
2147505415    UserMarker                 aten::empty                4480        0
2147508453    UserMarker                 aten::addcdiv_             15590       1280
2147490247    UserMarker                 aten::transpose            13210       0
2089          hipLaunchKernel                                       4090        1280
13336         hipLaunchKernel                                       5490        1280
2147496480    UserMarker                 autograd::engine::evaluat  36530       0
```

These tables(views) contain an entry for each function called.  You will likely want to aggregate them by apiName and/or args.

E.g. "select args, avg(cpu_time), avg(gpu_time) from callStack_inclusive_name group by args;"
or
```
import sqlite3
connection = sqlite3.connect("profile.rpd")
for row in connection.execute("select args, avg(cpu_time), avg(gpu_time) from callStack_inclusive_name group by args"):
        print(f"{row[0]}: {row[1]}, {row[2]}")
```


--------------------------------------------------------------------------------
## Stackframe recording
Setting `RPDT_STACKFRAMES=1` as an environment variable enables recording stack traces for every HIP API call. The data is recorded in the `rocpd_stackframe` table:
```
> select * from rocpd_stackframe limit 20;
id|api_ptr_id|depth|name_id
1|2|0|5
2|2|1|6
3|2|2|7
4|2|3|8
5|2|4|9
6|2|5|10
7|3|0|12
8|3|1|13
9|3|2|14
10|3|3|15
11|3|4|16
12|6|0|20
13|6|1|21
14|6|2|14
15|6|3|15
16|6|4|16
17|9|0|20
18|9|1|21
19|9|2|14
20|9|3|15
```
The `api_ptr_id` maps to the HIP API correlation ID, `depth` is the stack trace depth starting with 0, `name_id` is the stack frame mapping to `rocpd_string`.


--------------------------------------------------------------------------------

## Rpd_tracer Start/stop
Rpd_tracer recording is able to be toggled by the program being traced.  This allows for fine grained control over what is collected.
This is useful for:
- "trimming" files at collection time by simply not recording the warmup phase
- capturing managable amounts of data from long-running programs

#### From Python
There is an rpdTracerControl class available (as part of 'make install').  This class is able to load the rpd_tracer shared library, create an rpd file, and toggle recording.  It can be used as a context manager to do scoped recording, or to start/stop recording directly.

```
from rpdTracerControl import rpdTracerControl

# Optionally call this class method before creating first instance
rpdTracerControl.setFilename(name = "trace.rpd", append=False)

# Create first instance (this loads the profiler and creates the file)
profile = rpdTracerControl()

profile.start()
# recorded section
profile.stop()

with rpdTracerControl() as p:
  pass # recorded section

```

**Notes**:

The calls to start/stop recording are ref counted.  It is safe to use multiple recursive instances to record individual sections.  Do consider however, recording will remain active if any instance has it set on.

There are special considerations when using with python multiprocessing.  From the main process, before spawning any other processes:
  - Set the filename and append mode
  - Create one instance of the rpdTracerControl class

    *(This initializes an empty trace file (or appends existing) and sets up ENV variables used by the child processes.)*



#### From C/C++
This is possible but very much manual:
- create an empty rpd file (via python tools)  *See runTracer.sh for an example*
- suppress autostart of recording (via ENV)
- optionally set filename (via ENV)
- dlopen() librpd_tracer.so
- see "rpd_tracer.h" for symbols

Relevant ENV variables:
- RPDT_FILENAME = "./trace.rpd" 
- RPDT_AUTOSTART = 0



--------------------------------------------------------------------------------

## Schema V2
The RPD schema was changed to support copy and kernel args better. Previously kernel info and copy info were linked to an "op", i.e. gpu activity. Now they are linked to the api call that caused the execution.  This is required so that syncronous copies (ones that do not generate gpu ops) have a place to store args.

The new tables are:
- *rocpd_copyapi*
- *rocpd_kernelapi*

Both tables have a '*api_ptr_id*' column that can be used to join them to the api call (*rocpd_api*)  

The api call (*rocpd_api*) can be joined to the gpu op (*rocpd_op*) through the correlation id table (*rocpd_api_ops*)

The following views will be available in newly created files:
- **kernel**  
    Kernel info, with gpu timing
- **copy**  
    All copies, with cpu timing (i.e. the timespan of api call)
- **copyop**  
    Subset of copies that generated gpu ops, with gpu timing

There is now a key-value pair in (*rocpd_metadata*) with the schema version. E.g. ("schema_version", "2")
