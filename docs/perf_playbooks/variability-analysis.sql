.header on
.print "Registered GPUs and wall times"
select * from busy;

.print "Dropping old intermediate views if they exist"
drop view kernels_gpu0;
drop view kernels_gpu1;
drop view kernels_gpu2;
drop view kernels_gpu3;
drop view kernels_gpu4;
drop view kernels_gpu5;
drop view kernels_gpu6;
drop view kernels_gpu7;
drop view kernels_allgpu;
drop view kernel_minmaxavg;
drop view kernel_totalvariability;

.print "Types of kernel launches per GPU - ensure they are all the same! If not - project out types that mismatch"
select apiname, gpuId, count(*) from api join rocpd_api_ops on api.id = rocpd_api_ops.api_id join rocpd_op on rocpd_api_ops.op_id=rocpd_op.id group by apiname, gpuId;

.print "Creating GPU-resolved kernel views w/o collectives"
create view kernels_gpu0 as select apiName, s2.string as kernelName, (o.end-o.start) as duration, o.gpuId, row_number() over () as numRow from rocpd_op o inner join rocpd_string s2 on s2.id = o.description_id inner join rocpd_string s3 on s3.id = o.opType_id inner join rocpd_api_ops rao on rao.op_id=o.id inner join api on api.id = rao.api_id  where s3.string like 'KernelExecution' and s2.string not like '%ccl%' and o.gpuId = 0 order by o.start;
create view kernels_gpu1 as select apiName, s2.string as kernelName, (o.end-o.start) as duration, o.gpuId, row_number() over () as numRow from rocpd_op o inner join rocpd_string s2 on s2.id = o.description_id inner join rocpd_string s3 on s3.id = o.opType_id inner join rocpd_api_ops rao on rao.op_id=o.id inner join api on api.id = rao.api_id  where s3.string like 'KernelExecution' and s2.string not like '%ccl%' and o.gpuId = 1 order by o.start;
create view kernels_gpu2 as select apiName, s2.string as kernelName, (o.end-o.start) as duration, o.gpuId, row_number() over () as numRow from rocpd_op o inner join rocpd_string s2 on s2.id = o.description_id inner join rocpd_string s3 on s3.id = o.opType_id inner join rocpd_api_ops rao on rao.op_id=o.id inner join api on api.id = rao.api_id  where s3.string like 'KernelExecution' and s2.string not like '%ccl%' and o.gpuId = 2 order by o.start;
create view kernels_gpu3 as select apiName, s2.string as kernelName, (o.end-o.start) as duration, o.gpuId, row_number() over () as numRow from rocpd_op o inner join rocpd_string s2 on s2.id = o.description_id inner join rocpd_string s3 on s3.id = o.opType_id inner join rocpd_api_ops rao on rao.op_id=o.id inner join api on api.id = rao.api_id  where s3.string like 'KernelExecution' and s2.string not like '%ccl%' and o.gpuId = 3 order by o.start;
create view kernels_gpu4 as select apiName, s2.string as kernelName, (o.end-o.start) as duration, o.gpuId, row_number() over () as numRow from rocpd_op o inner join rocpd_string s2 on s2.id = o.description_id inner join rocpd_string s3 on s3.id = o.opType_id inner join rocpd_api_ops rao on rao.op_id=o.id inner join api on api.id = rao.api_id  where s3.string like 'KernelExecution' and s2.string not like '%ccl%' and o.gpuId = 4 order by o.start;
create view kernels_gpu5 as select apiName, s2.string as kernelName, (o.end-o.start) as duration, o.gpuId, row_number() over () as numRow from rocpd_op o inner join rocpd_string s2 on s2.id = o.description_id inner join rocpd_string s3 on s3.id = o.opType_id inner join rocpd_api_ops rao on rao.op_id=o.id inner join api on api.id = rao.api_id  where s3.string like 'KernelExecution' and s2.string not like '%ccl%' and o.gpuId = 5 order by o.start;
create view kernels_gpu6 as select apiName, s2.string as kernelName, (o.end-o.start) as duration, o.gpuId, row_number() over () as numRow from rocpd_op o inner join rocpd_string s2 on s2.id = o.description_id inner join rocpd_string s3 on s3.id = o.opType_id inner join rocpd_api_ops rao on rao.op_id=o.id inner join api on api.id = rao.api_id  where s3.string like 'KernelExecution' and s2.string not like '%ccl%' and o.gpuId = 6 order by o.start;
create view kernels_gpu7 as select apiName, s2.string as kernelName, (o.end-o.start) as duration, o.gpuId, row_number() over () as numRow from rocpd_op o inner join rocpd_string s2 on s2.id = o.description_id inner join rocpd_string s3 on s3.id = o.opType_id inner join rocpd_api_ops rao on rao.op_id=o.id inner join api on api.id = rao.api_id  where s3.string like 'KernelExecution' and s2.string not like '%ccl%' and o.gpuId = 7 order by o.start;

.print "Number of kernels per GPU track - ensure they are all the same!"
select count(*) as numKernelsGPU0 from kernels_gpu0;
select count(*) as numKernelsGPU1 from kernels_gpu1;
select count(*) as numKernelsGPU2 from kernels_gpu2;
select count(*) as numKernelsGPU3 from kernels_gpu3;
select count(*) as numKernelsGPU4 from kernels_gpu4;
select count(*) as numKernelsGPU5 from kernels_gpu5;
select count(*) as numKernelsGPU6 from kernels_gpu6;
select count(*) as numKernelsGPU7 from kernels_gpu7;

.print "Creating union of all non-collective kernels"
create view kernels_allgpu as select * from kernels_gpu0 union all select * from kernels_gpu1 union all select * from kernels_gpu2 union all select * from kernels_gpu3 union all select * from kernels_gpu4 union all select * from kernels_gpu5 union all select * from kernels_gpu6 union all select * from kernels_gpu7 order by numRow;

.print "Creating kernel-resolved statistics"
create view kernel_minmaxavg as select distinct numRow, avg(duration) over (partition by numRow) as avgDuration, min(duration) over (partition by numRow) as minDuration, max(duration) over (partition by numRow) as maxDuration from kernels_allgpu;


.print "Total variabilities in us"
create view kernel_totalvariability as select sum(maxDuration-minDuration) as maxToMinVariability, sum(maxDuration-avgDuration) as maxToAvgVariability from kernel_minmaxavg;
select * from kernel_totalvariability;

select max(GpuTime) as maxGPUTime from busy;
