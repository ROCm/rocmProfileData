--#
--# Calculate delay between kernel enqueue and kernel start
--#   Only considers kernels enqueues on an idle gpu (i.e. queueDepth = 0)
--#
--#   Some of the intermeditate views can be repurposed to display queueDepth at time of launch
--#
--#   
--#    
--#


--#  Flesh out joined api_op entries
create view if not exists api_op as select A.pid, A.tid, A.apiName, A.start as apiStart, A.end as apiEnd, B.gpuId, B.queueId, B.start as gpuStart, B.end as gpuEnd, kernelName from rocpd_api_ops Z join api A on A.id = Z.api_id join kernel B on B.id = Z.op_id order by A.start;

--#  View of queueDepth transitions (intermediate result, not useful in itself)
create view if not exists api_op_transition as select *, "1" as offset, apiStart as time from api_op UNION ALL select *, "-1" as offset, gpuEnd as time from api_op order by time;

--#  View of launch delays.  Includes those with queueDepth, which are being delayed because of kernel in front of them
create view if not exists launch_delay as select gpuId, apiName, kernelName, (gpuStart - apiEnd) as delay, sum(offset) over (partition by gpuId order by time) as queueDepth from api_op_transition where offset = "1";

--# Good to go.  Lets output only delays from when the gpu is idle (i.e. queueDepth = 1)
--#   Group by gpu and kernel name.  Is a certain gpu or kernel a problem spot?

.mode column
.header on

select gpuId, apiName, kernelName, count(gpuId) as count, min(delay) as min, max(delay) as max, avg(delay) avg from launch_delay where queueDepth = 1 group by gpuId, kernelName;

