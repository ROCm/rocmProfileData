--#
--# topDiff - Compare two traces and list the kernels in descending order of % change in execution time from one trace to another
--#   This is useful to figure out which kernels were the most impacted in terms of relative performance from one trace to another
--#   It is also important to pay attention to the Percentage columns as they indicate how much of the total execution time was taken
--#   up by each kernel. So, an entry with a high value for Percentage *and* PctDiff should be one of the top contributors to a perf drop
--#
--# NOTE: You'll need to update the rpd filenames below

ATTACH "<rpd filename 1>" as rpdA;
ATTACH "<rpd filename 2>" as rpdB;

SELECT *, (100.0*(B.Ave-A.Ave)/A.Ave) AS PctDiff FROM rdpA.top A JOIN rpdB.top B USING(Name) ORDER BY PctDiff DESC;
