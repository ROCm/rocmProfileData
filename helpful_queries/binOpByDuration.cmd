--#
--# Divide ops into bins based on execution time
--#   This will let you divide short, medium, long running, etc
--#
--#   Define your bins by altering the insert lines below
--#
--#   Overlapping bins are valid.  An item will be counted once for each
--#      bin it fits into.  E.g. bins like:  "< 10", "< 100", "< 1000"
--#


CREATE TEMPORARY TABLE bins ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "lower" integer NOT NULL, "upper" integer NOT NULL, "name" varchar(255) NOT NULL);
insert into bins(lower, upper, name) values (0, 10000, "< 10us");
insert into bins(lower, upper, name) values (10000, 25000, "10-25 us");
insert into bins(lower, upper, name) values (25000, 100000, "25-100 us");
insert into bins(lower, upper, name) values (100000, 1000000, "100-1000 us");
insert into bins(lower, upper, name) values (1000000, 1000000000, "> 1ms");

select B.name as Bin, count(*) as Count from op A join bins B on (A.end-A.start) < B.upper and (A.end-A.start) >= B.lower group by B.id;
