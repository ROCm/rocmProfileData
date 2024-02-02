CREATE VIEW api AS SELECT rocpd_api.id,pid,tid,start,end,A.string AS apiName, B.string AS args FROM rocpd_api INNER JOIN rocpd_string A ON A.id = rocpd_api.apiName_id INNER JOIN rocpd_string B ON B.id = rocpd_api.args_id;
CREATE VIEW op AS SELECT rocpd_op.id,gpuId,queueId,sequenceId,start,end,A.string AS description, B.string AS opType FROM rocpd_op INNER JOIN rocpd_string A ON A.id = rocpd_op.description_id INNER JOIN rocpd_string B ON B.id = rocpd_op.opType_id;
CREATE VIEW busy AS select A.gpuId, GpuTime, WallTime, GpuTime*1.0/WallTime as Busy from (select gpuId, sum(end-start) as GpuTime from rocpd_op group by gpuId) A INNER JOIN (select max(end) - min(start) as WallTime from rocpd_op);

create view ktop as select C.string as Name, count(C.string) as TotalCalls, sum(A.end-A.start) / 1000 as TotalDuration, (sum(A.end-A.start)/count(C.string))/ 1000 as Ave, sum(A.end-A.start) * 100.0 / (select sum(A.end-A.start) from rocpd_api A join rocpd_kernelapi B on B.api_ptr_id = A.id) as Percentage from rocpd_api A join rocpd_kernelapi B on B.api_ptr_id = A.id join rocpd_string C on C.id = B.kernelname_id group by Name order by TotalDuration desc;

create view top as select C.string as Name, count(C.string) as TotalCalls, sum(A.end-A.start) / 1000 as TotalDuration, (sum(A.end-A.start)/count(C.string))/ 1000 as Ave, sum(A.end-A.start) * 100.0 / (select sum(A.end-A.start) from rocpd_op A) as Percentage from (select opType_id as name_id, start, end from rocpd_op where description_id in (select id from rocpd_string where string='') union select description_id, start, end from rocpd_op where description_id not in (select id from rocpd_string where string='')) A join rocpd_string C on C.id = A.name_id group by Name order by TotalDuration desc;


-- Kernel ops with launch args
CREATE VIEW kernel AS SELECT B.id, gpuId, queueId, sequenceId, start, end, (end-start) AS duration, stream, gridX, gridY, gridz, workgroupX, workgroupY, workgroupZ, groupSegmentSize, privateSegmentSize, D.string AS kernelName FROM rocpd_api_ops A JOIN rocpd_op B on B.id = A.op_id JOIN rocpd_kernelapi C ON C.api_ptr_id = A.api_id JOIN rocpd_string D on D.id = kernelName_id;

-- All copies (api timing)
CREATE VIEW copy AS SELECT B.id, pid, tid, start, end, C.string AS apiName, stream, size, width, height, kind, dst, src, dstDevice, srcDevice, sync, pinned FROM rocpd_copyApi A JOIN rocpd_api B ON B.id = A.api_ptr_id JOIN rocpd_string C on C.id = B.apiname_id;

-- Async copies (op timing)
CREATE VIEW copyop AS SELECT B.id, gpuId, queueId, sequenceId, B.start, B.end, (B.end-B.start) AS duration, stream, size, width, height, kind, dst, src, dstDevice, srcDevice, sync, pinned, E.string AS apiName FROM rocpd_api_ops A JOIN rocpd_op B ON B.id = A.op_id JOIN rocpd_copyapi C ON C.api_ptr_id = A.api_id JOIN rocpd_api D on D.id = A.api_id JOIN rocpd_string E ON E.id = D.apiName_id;
