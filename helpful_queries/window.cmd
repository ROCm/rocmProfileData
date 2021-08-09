--#
--# Remove all rpd data that falls outside a desired window based on timestamps. 
--#   This can be useful to:
--#   1. create an rpd which doesn't have any warmup iterations, or contains data only within the desired time window, to perform further queries on
--#   2. dump a json trace which doesn't have any warmup iteration data, or data outside the desired time window
--#
--#   It might be convenient to insert roctx markers from the workload script to mark the start and end of the desired window. 
--#   To do this, use the hipScopedMarker python package (more details in ../hipMarker/README.txt) as follows:
--#   Example to start window after 10 iterations and end window after 15 iterations:
--#     from hipScopedMarker import hipScopedMarker
--#     while iteration < args.train_iters:
--#       if iteration == 10:
--#         hipScopedMarker.emitMarker("start_profile")
--#       if iteration == 15:
--#         hipScopedMarker.emitMarker("end_profile")
--#       train(...);


--# Apis
DELETE FROM rocpd_api WHERE start < (SELECT start FROM api WHERE apiname = "UserMarker" and args = "start_profile");
DELETE FROM rocpd_api WHERE start > (SELECT end FROM api WHERE apiname = "UserMarker" and args = "end_profile");

--# Ops
DELETE FROM rocpd_op WHERE start < (SELECT start FROM api WHERE apiname = "UserMarker" and args = "start_profile");
DELETE FROM rocpd_op WHERE start > (SELECT end FROM api WHERE apiname = "UserMarker" and args = "end_profile");

--# Cleanup
VACUUM
