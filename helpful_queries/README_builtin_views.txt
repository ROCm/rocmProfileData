RPD files created by the default tools will include some in-built views (for free!)
Views are stored queries.  They can be accessed like a normal table but are just
an alternate presentation.

These views are useful when interacing with the db directly via the sqlite3 cli.
To start try the following:

$ sqlite3 trace.rpd

sqlite> .mode column
sqlite> .header on
sqlite> .width 10 20 20 20 20    (to overide the default column width as you see fit)



API and OP views
----------------

In the rpd schema strings are stored seperately from the api and ops.  This makes it
hard to browse those events directly.  The 'api' and 'op' views join in the string table.

sqlite> select * from api limit 1;
id     pid       tid       start            end              apiName          args
-----  --------  --------  ---------------  ---------------  ---------------  ---------------
1      78436     78436     106107011171065  106107011184359  hipMalloc        size=0x3d0900

sqlite> select * from op limit 1;
id     gpuId     queueId   sequence  start         end           description   opType      
-----  --------  --------  --------  ------------  ------------  ------------  ---------------
1      0         0         0         106107047182  106107047229                CopyHostToDevice



TOP View
--------

This does an aggregation over all the ops and presents the ops that used the greatest gpu time.

sqlite> select * from top;
Name                          TotalCalls  TotalDuration  Ave         Percentage
----------------------------  ----------  -------------  ----------  ---------------
_Z13vector_squareIfEvPT_S1_m  499878      4719747        9           99.989909080028
CopyHostToDevice              1           476            476         0.0100909199719



KTOP View
---------

Similar to TOP except it only includes kernel ops.  This view requires the rocd_kernelop table
be populated.  Some collection tools may not do this.  The output may be empty.



BUSY View
---------

This reports average gpu usage over the run.  WARNING: this is only accurate on 'trimmed runs'
where any warmup has been removed.  Or possibly very long runs that amortize the warmpup.

sqlite> select * from busy;
gpuId       GpuTime     WallTime    Busy
----------  ----------  ----------  -----------------
0           4720223739  6453518297  0.731418665256478

