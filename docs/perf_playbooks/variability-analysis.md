## Variability analysis

### Preface
Variability is defined as the difference between kernel execution times of identical type, size, etc. on different GPUs. Variability is known to introduce a non-trivial performance penalty for multi-GPU workloads. This playbook uses a rpd database to quantify the performance impact arising from variability.

### Workflow
First, the `gpuIds` present in the rpd need to be queried. It is not always the case that they are 0-7, as can be seen in the example here:

```
sqlite> select * from busy;
gpuId|GpuTime|WallTime|Busy
2|81111298846|126018228685|0.643647349216032
3|74045687063|126018228685|0.587579176724404
4|84577265111|126018228685|0.671151038969232
5|85360254842|126018228685|0.677364344291569
6|85621957104|126018228685|0.67944104593014
7|85638792053|126018228685|0.679574637309544
8|85452706841|126018228685|0.678097984178153
9|86744546744|126018228685|0.688349198756237
```
In this case, GPU IDs 2-9 are populated.

Next, it needs to be checked whether the number of kernel launches on different GPUs is identical. 
```
sqlite> select apiname, gpuId, count(*) from api join rocpd_api_ops on api.id = rocpd_api_ops.api_id join rocpd_op on rocpd_api_ops.op_id=rocpd_op.id group by apiname, gpuId;
apiName|gpuId|count(*)
hipExtLaunchKernel|2|9853
hipExtLaunchKernel|3|9853
hipExtLaunchKernel|4|9853
hipExtLaunchKernel|5|9853
hipExtLaunchKernel|6|9853
hipExtLaunchKernel|7|9853
hipExtLaunchKernel|8|9853
hipExtLaunchKernel|9|9853
hipExtModuleLaunchKernel|2|123216
hipExtModuleLaunchKernel|3|123216
hipExtModuleLaunchKernel|4|123216
hipExtModuleLaunchKernel|5|123216
hipExtModuleLaunchKernel|6|123216
hipExtModuleLaunchKernel|7|123216
hipExtModuleLaunchKernel|8|123216
hipExtModuleLaunchKernel|9|123216
hipLaunchKernel|2|441421
hipLaunchKernel|3|441745
hipLaunchKernel|4|441483
hipLaunchKernel|5|441895
hipLaunchKernel|6|441836
hipLaunchKernel|7|441223
:hipLaunchKernel|8|441400
hipLaunchKernel|9|441384
hipMemcpy|2|18
hipMemcpy|3|18
hipMemcpy|4|18
hipMemcpy|5|18
hipMemcpy|6|18
hipMemcpy|7|18
hipMemcpy|8|18
hipMemcpy|9|18
...
```
In above case, the `hipLaunchKernel` counts on the GPUs are different and subsequently, they need to be projected out when intermediate views are created:
```
sqlite> create view kernels_gpu2 as select apiName, s2.string as kernelName, (o.end-o.start) as duration, o.gpuId, row_number() over () as numRow from rocpd_op o inner join rocpd_string s2 on s2.id = o.description_id inner join rocpd_string s3 on s3.id = o.opType_id inner join rocpd_api_ops rao on rao.op_id=o.id inner join api on api.id = rao.api_id  where s3.string like 'KernelExecution' and s2.string not like '%ccl%' and apiName not like 'hipLaunchKernel' and o.gpuId = 2 order by o.start;
sqlite> create view kernels_gpu3 as select apiName, s2.string as kernelName, (o.end-o.start) as duration, o.gpuId, row_number() over () as numRow from rocpd_op o inner join rocpd_string s2 on s2.id = o.description_id inner join rocpd_string s3 on s3.id = o.opType_id inner join rocpd_api_ops rao on rao.op_id=o.id inner join api on api.id = rao.api_id  where s3.string like 'KernelExecution' and s2.string not like '%ccl%' and apiName not like 'hipLaunchKernel' and o.gpuId = 3 order by o.start;
sqlite> create view kernels_gpu4 as select apiName, s2.string as kernelName, (o.end-o.start) as duration, o.gpuId, row_number() over () as numRow from rocpd_op o inner join rocpd_string s2 on s2.id = o.description_id inner join rocpd_string s3 on s3.id = o.opType_id inner join rocpd_api_ops rao on rao.op_id=o.id inner join api on api.id = rao.api_id  where s3.string like 'KernelExecution' and s2.string not like '%ccl%' and apiName not like 'hipLaunchKernel' and o.gpuId = 4 order by o.start;
...
sqlite> create view kernels_gpu9 as select apiName, s2.string as kernelName, (o.end-o.start) as duration, o.gpuId, row_number() over () as numRow from rocpd_op o inner join rocpd_string s2 on s2.id = o.description_id inner join rocpd_string s3 on s3.id = o.opType_id inner join rocpd_api_ops rao on rao.op_id=o.id inner join api on api.id = rao.api_id  where s3.string like 'KernelExecution' and s2.string not like '%ccl%' and apiName not like 'hipLaunchKernel' and o.gpuId = 9 order by o.start;

```
This forms views for every GPU that contain the `apiName` (e.g., `hipExtModuleLaunchKernel`), the `kernelName`, the kernel `duration`, the `gpuId`, and assigns a `numRow` after sorting by GPU start time in ascending order.

Note that above needs to be adjusted if GPU IDs other than 2-9 are used in the rpd. Also this projects `hipLaunchKernel` launched kernels out.

Confirmation that all tracks contain the same number of kernel launches now:
```
sqlite> select count(*) as numKernelsGPU2 from kernels_gpu2;
numKernelsGPU2
123216
sqlite> select count(*) as numKernelsGPU3 from kernels_gpu3;
numKernelsGPU3
123216
sqlite> select count(*) as numKernelsGPU4 from kernels_gpu4;
numKernelsGPU4
123216
sqlite> select count(*) as numKernelsGPU5 from kernels_gpu5;
numKernelsGPU5
123216
sqlite> select count(*) as numKernelsGPU6 from kernels_gpu6;
numKernelsGPU6
123216
sqlite> select count(*) as numKernelsGPU7 from kernels_gpu7;
numKernelsGPU7
123216
sqlite> select count(*) as numKernelsGPU8 from kernels_gpu8;
numKernelsGPU8
123216
sqlite> select count(*) as numKernelsGPU9 from kernels_gpu9;
numKernelsGPU9
123216
```
In this instance, 123216 kernels are tracked on each track. If the counts are not matched at this point, the intermediate views that were created are wrong.

Next, the different GPU tracks are merged and sorted by `numRow`:
```
sqlite> create view kernels_allgpu as select * from kernels_gpu2 union all select * from kernels_gpu3 union all select * from kernels_gpu4 union all select * from kernels_gpu5 union all select * from kernels_gpu6 union all select * from kernels_gpu7 union all select * from kernels_gpu8 union all select * from kernels_gpu9 order by numRow;
```

Now, the combined track can be collapsed to get the average, minimal, and maximal kernel durations assuming that idential `numRow` correspond to identical kernels:
```
sqlite> create view kernel_minmaxavg as select distinct numRow, avg(duration) over (partition by numRow) as avgDuration, min(duration) over (partition by numRow) as minDuration, max(duration) over (partition by numRow) as maxDuration from kernels_allgpu;
```

Finally, the difference between minimal or average duration and maximal duration can be summed up:
```
sqlite> create view kernel_totalvariability as select sum(maxDuration-minDuration) as maxToMinVariability, sum(maxDuration-avgDuration) as maxToAvgVariability from kernel_minmaxavg;
sqlite> select * from kernel_totalvariability;
maxToMinVariability|maxToAvgVariability
8520963123|4146629083.375
sqlite> select max(GpuTime) as maxGPUTime from busy;
maxGPUTime
86744546744
```
In this case, the variability is estimated to be 8520963123 us between best and worst kernel executions and 4146629083 us between average and worst kernel executions - or 9.8% and 4.8%, respectively.


### Summary
The performance impact arising from variability can be estimated using the above protocol. It is important to note that the SQL queries are dependent on and must be adjusted for different GPU setups (e.g., 4 vs 8 GPUs, GPU IDs 0-7 vs 2-9) and tracing databases must be cleaned to arrive at intermediate tracks that have the same kernel executions in the same order across all GPUs. Two example workflows are available for [matched](variability-analysis.sql) and [unmatched](variability-analysis_nolaunch.sql) rpds.
