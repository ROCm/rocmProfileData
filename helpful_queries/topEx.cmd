--#
--# TopEx - Output 'top' apis (host/CPU) and ops (GPU), exclusive.
--#   Time from nested calls is only attributed to the callee, not the caller.
--#

.mode column
.header on
.width 40 12 12 12 12

--# Apis
CREATE TABLE temp.seq ("id" integer NOT NULL PRIMARY KEY, "srcId" integer, "ts" integer NOT NULL, "trans_type" integer NOT NULL, "total" integer, "topmost" integer);
INSERT INTO temp.seq(srcId, ts, trans_type) SELECT id, start AS ts, '1' FROM rocpd_api UNION ALL SELECT id, end, '-1' FROM rocpd_api ORDER BY ts ASC;
UPDATE temp.seq SET total = 1 where id = 1;
UPDATE temp.seq SET total = temp.seq.trans_type + (SELECT total FROM temp.seq AS A WHERE A.id = temp.seq.id - 1) WHERE id > 1;

UPDATE temp.seq SET topmost = (SELECT srcId) WHERE trans_type=1;
UPDATE temp.seq SET topmost = (SELECT topmost FROM temp.seq AS A WHERE A.total = temp.seq.total AND A.id < temp.seq.id ORDER BY id DESC LIMIT 1) WHERE trans_type=-1;

CREATE VIEW temp.exclusive_api AS select A.id, C.pid, C.tid, A.ts as start, B.ts as end, C.apiName_id, C.args_id from temp.seq A join temp.seq B on B.id = a.id + 1 join rocpd_api C on C.id = A.topmost;

--# raw ranges
--# select A.topmost, A.ts as start, B.ts as end, B.ts - A.ts as Duration from temp.seq A join temp.seq B on B.id = a.id + 1
--# select A.id, C.pid, C.tid, A.ts as start, B.ts as end, C.apiName_id, C.args_id from temp.seq A join temp.seq B on B.id = a.id + 1 join rocpd_api C on C.id = A.topmost;


select A.string as ApiName, count(A.string) as TotalSections, sum(B.end - B.start) / 1000 as totalDuration, (sum(B.end-B.start)/count(A.string)) / 1000 as Ave, sum(B.end-B.start) * 100 / (select sum(end-start) from temp.exclusive_api) as Percentage from temp.exclusive_api B join rocpd_string A on A.id = B.apiName_id group by ApiName order by TotalDuration desc;

--#Ops
DROP TABLE temp.seq;
CREATE TABLE temp.seq ("id" integer NOT NULL PRIMARY KEY, "srcId" integer, "ts" integer NOT NULL, "trans_type" integer NOT NULL, "total" integer, "topmost" integer);
INSERT INTO temp.seq(srcId, ts, trans_type) SELECT id, start AS ts, '1' FROM rocpd_op UNION ALL SELECT id, end, '-1' FROM rocpd_op ORDER BY ts ASC;
UPDATE temp.seq SET total = 1 where id = 1;
UPDATE temp.seq SET total = temp.seq.trans_type + (SELECT total FROM temp.seq AS A WHERE A.id = temp.seq.id - 1) WHERE id > 1;

UPDATE temp.seq SET topmost = (SELECT srcId) WHERE trans_type=1;
UPDATE temp.seq SET topmost = (SELECT topmost FROM temp.seq AS A WHERE A.total = temp.seq.total AND A.id < temp.seq.id ORDER BY id DESC LIMIT 1) WHERE trans_type=-1;

CREATE VIEW temp.exclusive_op AS select A.id, C.gpuId, C.queueId, C.sequenceId, C.completionSignal, A.ts as start, B.ts as end, C.description_id, C.opType_id from temp.seq A join temp.seq B on B.id = a.id + 1 join rocpd_op C on C.id = A.topmost;

select A.string as KernelName, count(A.string) as TotalSections, sum(B.end - B.start) / 1000 as totalDuration, (sum(B.end-B.start)/count(A.string)) / 1000 as Ave, sum(B.end-B.start) * 100 / (select sum(end-start) from temp.exclusive_op) as Percentage from temp.exclusive_op B join rocpd_string A on A.id = B.description_id group by KernelName order by TotalDuration desc;


