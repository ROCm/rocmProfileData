--#
--# Calculate delay between adjacently running kernels
--#   Only considers kernels when the gpu is busy (i.e. queueDepth > 0)
--#
--#
--#
--#
--#
--#


--#  Flesh out joined api_op entries
create view if not exists api_op as select A.pid, A.tid, A.apiName, A.start as apiStart, A.end as apiEnd, B.gpuId, B.queueId, B.start as gpuStart, B.end as gpuEnd, kernelName from rocpd_api_ops Z join api A on A.id = Z.api_id join kernel B on B.id = Z.op_id order by A.start;

--#  View of queueDepth transitions (intermediate result, not useful in itself)
create view if not exists api_op_transition as select *, "1" as offset, apiStart as time from api_op UNION ALL select *, "-1" as offset, gpuEnd as time from api_op order by time;


--#  View of kernel timing with queueDepth included
create view if not exists kernel_timing as select gpuId, gpuStart, gpuEnd, sum(offset) over (partition by gpuId order by time) as queueDepth from api_op_transition where offset = "1";

--#  View of duration and kernel delay from pervious kernel completion
create view if not exists kernel_delay as select gpuId, gpuStart - LAG(gpuEnd) over (partition by gpuid order by gpuEnd) as kkDelay, gpuEnd - gpuStart as duration from kernel_timing where queueDepth > 1;

--# Output the average kernel time and inter-kernel delays per gpu

.mode column
.header on

select gpuId, avg(kkDelay) as avgDelay, avg(duration) as avgDuration, avg(kkDelay) * 100 / (avg(kkDelay) + avg(duration)) as percentGapLoss from kernel_delay group by gpuId;
