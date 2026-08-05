# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import bisect
import json
import sqlite3
import argparse
import pathlib

_s = json.dumps


class GpuFrame:
    def __init__(self):
        self.id = 0
        self.name = ''
        self.start = 0
        self.end = 0
        self.gpus = []
        self.totalOps = 0

    @classmethod
    def from_op(cls, frame_id, frame_name, op_start, op_end, gpu_id, queue_id):
        f = cls()
        f.id = frame_id
        f.name = frame_name
        f.start = op_start
        f.end = op_end
        f.gpus.append((gpu_id, queue_id))
        f.totalOps = 1
        return f


def _ensure_schema_compat(connection):
    try:
        row = connection.execute(
            "SELECT value FROM rocpd_metadata WHERE tag = 'schema_version'"
        ).fetchone()
        schema_version = int(row[0]) if row else None
    except sqlite3.OperationalError:
        schema_version = None

    if schema_version is None or schema_version <= 2:
        connection.execute(
            "CREATE TEMPORARY VIEW rocpd_ustring AS SELECT * FROM rocpd_string"
        )
        print(f"Schema: v{schema_version}")


def _get_time_range(connection):
    row = connection.execute(
        "SELECT MIN(start), MAX(end) FROM rocpd_api"
    ).fetchone()
    if row[0] is None:
        raise Exception("Trace file is empty.")
    return row[0], row[1]


def _parse_time_bounds(min_time, max_time, start_arg, end_arg):
    start_us = min_time / 1000
    end_us = max_time / 1000

    if start_arg:
        if "%" in start_arg:
            pct = int(start_arg.replace("%", ""))
            start_us = ((max_time - min_time) * (pct / 100) + min_time) / 1000
        else:
            start_us = int(start_arg)

    if end_arg:
        if "%" in end_arg:
            pct = int(end_arg.replace("%", ""))
            end_us = ((max_time - min_time) * (pct / 100) + min_time) / 1000
        else:
            end_us = int(end_arg)

    return start_us, end_us


def _build_range_filter(table_prefix, start_time=None, end_time=None):
    col = f"{table_prefix}.start" if table_prefix else "start"
    conditions = []
    params = []
    if start_time is not None:
        conditions.append(f"{col}/1000 >= ?")
        params.append(start_time)
    if end_time is not None:
        conditions.append(f"{col}/1000 <= ?")
        params.append(end_time)
    if not conditions:
        return "", ()
    return " WHERE " + " AND ".join(conditions), tuple(params)


def _load_ops_by_gpu(connection, range_op, params_op):
    ops_by_gpu = {}
    query = "SELECT gpuId, queueId, start, end FROM rocpd_op" + range_op + " ORDER BY start"
    for row in connection.execute(query, params_op):
        key = (row[0], row[1])
        if key not in ops_by_gpu:
            ops_by_gpu[key] = ([], [])
        ops_by_gpu[key][0].append(row[2])
        ops_by_gpu[key][1].append(row[3])
    return ops_by_gpu


def _gpu_user_annotations(ops_by_gpu, scopes):
    results = []
    for api_id, api_name, scope_start, scope_end in scopes:
        for (gpu_id, queue_id), (starts, ends) in ops_by_gpu.items():
            lo = bisect.bisect_left(starts, scope_start)
            hi = bisect.bisect_right(starts, scope_end)
            if lo >= hi:
                continue
            gpu_start = starts[lo] / 1000.0
            gpu_end = max(ends[lo:hi]) / 1000.0
            results.append((api_id, api_name, gpu_id, queue_id, gpu_start, gpu_end - gpu_start))
    return results


def generate_rpd_json(connection, outfile, start=None, end=None, trace_format="object", verbose=False):
    _ensure_schema_compat(connection)

    write = outfile.write

    if trace_format == "object":
        write('{"traceEvents": ')
    write("[ {}\n")

    v3 = _has_v3_schema(connection)

    # GPU process metadata
    for row in connection.execute("SELECT DISTINCT gpuId FROM rocpd_op"):
        gpu_id = row[0]
        write(',{"name":"process_name","ph":"M","pid":"%s","args":{"name":"GPU%s"}}\n' % (gpu_id, gpu_id))
        write(',{"name":"process_labels","ph":"M","pid":"%s","args":{"labels":"GPU %s"}}\n' % (gpu_id, gpu_id))
        write(',{"name":"process_sort_index","ph":"M","pid":"%s","args":{"sort_index":"%s"}}\n' % (gpu_id, gpu_id + 1000000))

    # CPU process metadata
    for row in connection.execute("SELECT DISTINCT pid FROM rocpd_api"):
        pid = row[0]
        write(',{"name":"process_labels","ph":"M","pid":"%s","args":{"labels":"CPU"}}\n' % pid)

    # HIP API thread metadata
    for row in connection.execute("SELECT DISTINCT pid, tid FROM rocpd_api"):
        write(',{"name":"thread_name","ph":"M","pid":"%s","tid":"%s","args":{"name":"Hip %s"}}\n' % (row[0], row[1], row[1]))
        write(',{"name":"thread_sort_index","ph":"M","pid":"%s","tid":"%s","args":{"sort_index":"%s"}}\n' % (row[0], row[1], row[1] * 2))

    # Time range
    min_time, max_time = _get_time_range(connection)
    if verbose:
        print("Timestamps:")
        print(f"\t    first: \t{min_time/1000} us")
        print(f"\t     last: \t{max_time/1000} us")
        print(f"\t duration: \t{(max_time-min_time) / 1000000000} seconds")

    start_us, end_us = _parse_time_bounds(min_time, max_time, start, end)

    filter_start = start_us if start else None
    filter_end = end_us if end else None

    range_api, params_api = _build_range_filter("rocpd_api", filter_start, filter_end)
    range_op, params_op = _build_range_filter("rocpd_op", filter_start, filter_end)
    range_monitor, params_monitor = _build_range_filter("", filter_start, filter_end)

    if verbose:
        print(f"\nFilter: {range_api}")
        print(f"Output duration: {(end_us-start_us)/1000000} seconds")

    # GPU ops with optional kernel launch config
    has_kernelapi = _table_exists(connection, "rocpd_kernelapi")
    if has_kernelapi:
        query = (
            "SELECT A.string AS optype, B.string AS description, gpuId, queueId, "
            "rocpd_op.start/1000.0, (rocpd_op.end - rocpd_op.start) / 1000.0, "
            "K.gridX, K.gridY, K.gridZ, K.workgroupX, K.workgroupY, K.workgroupZ, "
            "K.groupSegmentSize "
            "FROM rocpd_op "
            "INNER JOIN rocpd_string A ON A.id = rocpd_op.opType_id "
            "INNER JOIN rocpd_string B ON B.id = rocpd_op.description_id "
            "LEFT JOIN rocpd_api_ops ON rocpd_api_ops.op_id = rocpd_op.id "
            "LEFT JOIN rocpd_kernelapi K ON K.api_ptr_id = rocpd_api_ops.api_id"
            + range_op
        )
        for row in connection.execute(query, params_op):
            name = row[0] if len(row[1]) == 0 else row[1]
            if row[6] is not None:
                write(',{"pid":"%s","tid":"%s","name":%s,"ts":%s,"dur":%s,"ph":"X","args":{"desc":%s,"grid":[%s,%s,%s],"block":[%s,%s,%s],"shared memory":%s}}\n' % (row[2], row[3], _s(name), row[4], row[5], _s(row[0]), row[6], row[7], row[8], row[9], row[10], row[11], row[12]))
            else:
                write(',{"pid":"%s","tid":"%s","name":%s,"ts":%s,"dur":%s,"ph":"X","args":{"desc":%s}}\n' % (row[2], row[3], _s(name), row[4], row[5], _s(row[0])))
    else:
        query = (
            "SELECT A.string AS optype, B.string AS description, gpuId, queueId, "
            "rocpd_op.start/1000.0, (rocpd_op.end - rocpd_op.start) / 1000.0 "
            "FROM rocpd_op "
            "INNER JOIN rocpd_string A ON A.id = rocpd_op.opType_id "
            "INNER JOIN rocpd_string B ON B.id = rocpd_op.description_id"
            + range_op
        )
        for row in connection.execute(query, params_op):
            name = row[0] if len(row[1]) == 0 else row[1]
            write(',{"pid":"%s","tid":"%s","name":%s,"ts":%s,"dur":%s,"ph":"X","args":{"desc":%s}}\n' % (row[2], row[3], _s(name), row[4], row[5], _s(row[0])))

    # Graph executions
    try:
        range_graph, params_graph = _build_range_filter("C", filter_start, filter_end)
        query = (
            "SELECT graphExec, gpuId, queueId, min(start)/1000.0, "
            "(max(end)-min(start))/1000.0, count(*) "
            "FROM rocpd_graphLaunchapi A "
            "JOIN rocpd_api_ops B ON B.api_id = A.api_ptr_id "
            "JOIN rocpd_op C ON C.id = B.op_id"
            + range_graph
            + " GROUP BY api_ptr_id"
        )
        for row in connection.execute(query, params_graph):
            write(',{"pid":"%s","tid":"%s","name":"Graph %s","ts":%s,"dur":%s,"ph":"X","args":{"kernels":"%s"}}\n' % (row[1], row[2], row[0], row[3], row[4], row[5]))
    except sqlite3.OperationalError:
        pass

    # API calls
    query = (
        "SELECT A.string AS apiName, B.string AS args, pid, tid, "
        "rocpd_api.start/1000.0, (rocpd_api.end - rocpd_api.start) / 1000.0, "
        "(rocpd_api.end != rocpd_api.start) AS has_duration "
        "FROM rocpd_api "
        "INNER JOIN rocpd_string A ON A.id = rocpd_api.apiName_id "
        "INNER JOIN rocpd_ustring B ON B.id = rocpd_api.args_id"
        + range_api
        + " ORDER BY rocpd_api.id"
    )
    for row in connection.execute(query, params_api):
        if row[0] == "UserMarker":
            desc = _s(row[1])
            if row[6] == 0:
                write(',{"pid":"%s","tid":"%s","name":%s,"ts":%s,"ph":"i","s":"p","args":{"desc":%s}}\n' % (row[2], row[3], desc, row[4], desc))
            else:
                write(',{"pid":"%s","tid":"%s","name":%s,"ts":%s,"dur":%s,"ph":"X","args":{"desc":%s}}\n' % (row[2], row[3], desc, row[4], row[5], desc))
        else:
            write(',{"pid":"%s","tid":"%s","name":%s,"ts":%s,"dur":%s,"ph":"X","args":{"desc":%s}}\n' % (row[2], row[3], _s(row[0]), row[4], row[5], _s(row[1])))

    # API->Op flow linkage
    query = (
        "SELECT rocpd_api_ops.id, pid, tid, gpuId, queueId, "
        "rocpd_api.end/1000.0 - 2, rocpd_op.start/1000.0 "
        "FROM rocpd_api_ops "
        "INNER JOIN rocpd_api ON rocpd_api_ops.api_id = rocpd_api.id "
        "INNER JOIN rocpd_op ON rocpd_api_ops.op_id = rocpd_op.id"
        + range_api
    )
    for row in connection.execute(query, params_api):
        fromtime = row[5] if row[5] < row[6] else row[6]
        write(',{"pid":"%s","tid":"%s","cat":"api_op","name":"api_op","ts":%s,"id":"%s","ph":"s"}\n' % (row[1], row[2], fromtime, row[0]))
        write(',{"pid":"%s","tid":"%s","cat":"api_op","name":"api_op","ts":%s,"id":"%s","ph":"f","bp":"e"}\n' % (row[3], row[4], row[6], row[0]))

    # Counters - find T_end
    t_end = 0
    for row in connection.execute(
        "SELECT max(end)/1000 FROM "
        "(SELECT end FROM rocpd_api UNION ALL SELECT end FROM rocpd_op)"
    ):
        t_end = int(row[0])
    if end:
        t_end = end_us

    # Per-GPU queue depth and idle counters
    gpu_ids = [row[0] for row in connection.execute(
        "SELECT DISTINCT gpuId FROM rocpd_op"
    )]

    for gpu_id in gpu_ids:
        range_op_clause, range_op_params = _build_range_filter(
            "rocpd_op", filter_start, filter_end
        )
        gpu_params = (gpu_id,) + range_op_params + (gpu_id,) + range_op_params
        query = (
            "SELECT * FROM ("
            "SELECT rocpd_api.start/1000.0 AS ts, '1' "
            "FROM rocpd_api_ops "
            "INNER JOIN rocpd_api ON rocpd_api_ops.api_id = rocpd_api.id "
            "INNER JOIN rocpd_op ON rocpd_api_ops.op_id = rocpd_op.id "
            "AND rocpd_op.gpuId = ?"
            + range_op_clause
            + " UNION ALL "
            "SELECT rocpd_op.end/1000.0, '-1' "
            "FROM rocpd_api_ops "
            "INNER JOIN rocpd_api ON rocpd_api_ops.api_id = rocpd_api.id "
            "INNER JOIN rocpd_op ON rocpd_api_ops.op_id = rocpd_op.id "
            "AND rocpd_op.gpuId = ?"
            + range_op_clause
            + ") ORDER BY ts"
        )
        depth = 0
        idle = 1
        for row in connection.execute(query, gpu_params):
            delta = int(row[1])
            if idle and delta > 0:
                idle = 0
                write(',{"pid":"%s","name":"Idle","ph":"C","ts":%s,"args":{"idle":%s}}\n' % (gpu_id, row[0], idle))
            if depth == 1 and delta < 0:
                idle = 1
                write(',{"pid":"%s","name":"Idle","ph":"C","ts":%s,"args":{"idle":%s}}\n' % (gpu_id, row[0], idle))
            depth += delta
            write(',{"pid":"%s","name":"QueueDepth","ph":"C","ts":%s,"args":{"depth":%s}}\n' % (gpu_id, row[0], depth))
        if t_end > 0:
            write(',{"pid":"%s","name":"Idle","ph":"C","ts":%s,"args":{"idle":%s}}\n' % (gpu_id, t_end, idle))
            write(',{"pid":"%s","name":"QueueDepth","ph":"C","ts":%s,"args":{"depth":%s}}\n' % (gpu_id, t_end, depth))

    # SMI monitor counters
    try:
        query = (
            "SELECT deviceId, monitorType, start/1000.0, value "
            "FROM rocpd_monitor" + range_monitor
        )
        for row in connection.execute(query, params_monitor):
            write(',{"pid":"%s","name":"%s","ph":"C","ts":%s,"args":{"%s":%s}}\n' % (row[0], row[1], row[2], row[1], row[3]))
        query = (
            "SELECT DISTINCT deviceId, monitorType, max(end)/1000.0, value "
            "FROM rocpd_monitor" + range_monitor
            + " GROUP BY deviceId, monitorType"
        )
        for row in connection.execute(query, params_monitor):
            write(',{"pid":"%s","name":"%s","ph":"C","ts":%s,"args":{"%s":%s}}\n' % (row[0], row[1], row[2], row[1], row[3]))
    except sqlite3.OperationalError:
        if verbose:
            print("Did not find SMI data")

    # UserMarker / user_scope GPU frames
    stacks = {}
    current_frame = {}

    range_api_clause, range_api_params = _build_range_filter(
        "rocpd_api", filter_start, filter_end
    )
    frame_params = range_api_params + range_api_params + range_api_params

    if v3:
        marker_filter = (
            "INNER JOIN rocpd_string _D ON _D.id = rocpd_api.domain_id "
            "AND _D.string = 'torch' "
        )
        marker_label_join = "INNER JOIN rocpd_string B ON B.id = rocpd_api.apiName_id "
    else:
        marker_filter = (
            "INNER JOIN rocpd_string A ON A.id = rocpd_api.apiName_id "
            "AND A.string = 'UserMarker' "
        )
        marker_label_join = "INNER JOIN rocpd_ustring B ON B.id = rocpd_api.args_id "

    query = (
        "SELECT '0', start/1000.0, pid, tid, B.string AS label, "
        "'','','','' "
        "FROM rocpd_api "
        + marker_filter
        + marker_label_join
        + "AND rocpd_api.start/1000.0 != rocpd_api.end/1000.0"
        + range_api_clause
        + " UNION ALL "
        "SELECT '1', end/1000.0, pid, tid, B.string AS label, "
        "'','','','' "
        "FROM rocpd_api "
        + marker_filter
        + marker_label_join
        + "AND rocpd_api.start/1000.0 != rocpd_api.end/1000.0"
        + range_api_clause
        + " UNION ALL "
        "SELECT '2', rocpd_api.start/1000.0, pid, tid, '' AS label, "
        "gpuId, queueId, rocpd_op.start/1000.0, rocpd_op.end/1000.0 "
        "FROM rocpd_api_ops "
        "INNER JOIN rocpd_api ON rocpd_api_ops.api_id = rocpd_api.id "
        "INNER JOIN rocpd_op ON rocpd_api_ops.op_id = rocpd_op.id"
        + range_api_clause
        + " ORDER BY start/1000.0 ASC"
    )

    for row in connection.execute(query, frame_params):
        key = (row[2], row[3])
        if row[0] == '0':
            if key not in stacks:
                stacks[key] = []
            stacks[key].append((row[1], row[4]))

        elif row[0] == '1':
            stacks[key].pop()

        elif row[0] == '2':
            if key in stacks and len(stacks[key]) > 0:
                frame = stacks[key][-1]
                if key not in current_frame:
                    current_frame[key] = GpuFrame.from_op(
                        frame[0], frame[1], row[7], row[8], row[5], row[6]
                    )
                else:
                    gpu_frame = current_frame[key]
                    if (gpu_frame.id == frame[0]
                            and gpu_frame.name == frame[1]
                            and (abs(row[7] - gpu_frame.end) < 200
                                 or abs(gpu_frame.start - row[8]) < 200)):
                        if row[7] < gpu_frame.start:
                            gpu_frame.start = row[7]
                        if row[8] > gpu_frame.end:
                            gpu_frame.end = row[8]
                        if (row[5], row[6]) not in gpu_frame.gpus:
                            gpu_frame.gpus.append((row[5], row[6]))
                        gpu_frame.totalOps += 1
                    else:
                        for dest in gpu_frame.gpus:
                            write(',{"pid":"%s","tid":"%s","name":%s,"ts":%s,"dur":%s,"ph":"X","args":{"desc":"UserMarker frame: %s ops"}}\n' % (dest[0], dest[1], _s(gpu_frame.name), gpu_frame.start - 1, gpu_frame.end - gpu_frame.start + 1, gpu_frame.totalOps))
                        current_frame[key] = GpuFrame.from_op(
                            frame[0], frame[1], row[7], row[8], row[5], row[6]
                        )

    # fwdbwd flow events: forward op -> backward op linked by seq (v3 only)
    if v3:
        fwd_seq_events = {}
        bwd_seq_events = []
        for row in connection.execute(
            "SELECT rocpd_api.start/1000.0, pid, tid, "
            "B.string AS category, "
            "json_extract(D.string, '$.seq') AS seq "
            "FROM rocpd_api "
            "INNER JOIN rocpd_string B ON B.id = rocpd_api.category_id "
            "INNER JOIN rocpd_ustring D ON D.id = rocpd_api.args_id "
            "WHERE B.string IN ('function', 'backward_function') "
            "AND D.string LIKE '{%' "
            "AND json_extract(D.string, '$.seq') >= 0"
            + (range_api.replace(" WHERE ", " AND ", 1) if range_api else "")
            + " ORDER BY rocpd_api.id"
        ):
            seq = row[4]
            if row[3] == 'function':
                fwd_seq_events[seq] = (row[0], row[1], row[2])
            else:
                bwd_seq_events.append((seq, row[0], row[1], row[2]))

        flow_id = 0
        for seq, bwd_ts, bwd_pid, bwd_tid in bwd_seq_events:
            if seq in fwd_seq_events:
                fwd_ts, fwd_pid, fwd_tid = fwd_seq_events[seq]
                flow_id += 1
                write(',{"pid":"%s","tid":"%s","cat":"fwdbwd","name":"fwdbwd","ts":%s,"id":"%s","ph":"s"}\n' % (fwd_pid, fwd_tid, fwd_ts, flow_id))
                write(',{"pid":"%s","tid":"%s","cat":"fwdbwd","name":"fwdbwd","ts":%s,"id":"%s","ph":"f","bp":"e"}\n' % (bwd_pid, bwd_tid, bwd_ts, flow_id))

    write("]\n")
    if trace_format == "object":
        write("}\n")


_KINETO_DOMAIN_CAT = {
    ('hip', ''): 'cuda_runtime',
    ('torch', 'function'): 'cpu_op',
    ('torch', 'backward_function'): 'cpu_op',
    ('torch', 'user_scope'): 'user_annotation',
}


def _has_v3_schema(connection):
    try:
        connection.execute("SELECT domain_id FROM rocpd_api LIMIT 1")
        return True
    except sqlite3.OperationalError:
        return False


def _table_exists(connection, table_name):
    try:
        connection.execute("SELECT 1 FROM %s LIMIT 1" % table_name)
        return True
    except sqlite3.OperationalError:
        return False


def generate_kineto_json(connection, outfile, start=None, end=None, trace_format="object", verbose=False):
    _ensure_schema_compat(connection)
    write = outfile.write

    v3 = _has_v3_schema(connection)
    has_kernelapi = _table_exists(connection, "rocpd_kernelapi")

    write('{"schemaVersion":1,"displayTimeUnit":"ms","traceEvents":[\n{}')

    # Time range
    min_time, max_time = _get_time_range(connection)
    start_us, end_us = _parse_time_bounds(min_time, max_time, start, end)

    filter_start = start_us if start else None
    filter_end = end_us if end else None

    range_api, params_api = _build_range_filter("rocpd_api", filter_start, filter_end)
    range_op, params_op = _build_range_filter("rocpd_op", filter_start, filter_end)

    if verbose:
        print("Timestamps:")
        print(f"\t    first: \t{min_time/1000} us")
        print(f"\t     last: \t{max_time/1000} us")
        print(f"\t duration: \t{(max_time-min_time) / 1000000000} seconds")
        print(f"\nFilter: {range_api}")
        print(f"Output duration: {(end_us-start_us)/1000000} seconds")

    # Collect distinct pids and (pid,tid) pairs for metadata
    cpu_pids = set()
    cpu_tids = set()
    for row in connection.execute("SELECT DISTINCT pid, tid FROM rocpd_api"):
        cpu_pids.add(row[0])
        cpu_tids.add((row[0], row[1]))

    gpu_streams = set()
    for row in connection.execute("SELECT DISTINCT gpuId, queueId FROM rocpd_op"):
        gpu_streams.add((row[0], row[1]))

    gpu_ids = sorted({g[0] for g in gpu_streams})
    ts_meta = min_time / 1000.0

    # CPU process metadata
    for pid in sorted(cpu_pids):
        write(',{"name":"process_name","ph":"M","ts":%s,"pid":%s,"tid":0,"args":{"name":"python"}}\n' % (ts_meta, pid))
        write(',{"name":"process_labels","ph":"M","ts":%s,"pid":%s,"tid":0,"args":{"labels":"CPU"}}\n' % (ts_meta, pid))
        write(',{"name":"process_sort_index","ph":"M","ts":%s,"pid":%s,"tid":0,"args":{"sort_index":%s}}\n' % (ts_meta, pid, pid))

    # GPU process metadata
    for gpu_id in gpu_ids:
        write(',{"name":"process_name","ph":"M","ts":%s,"pid":%s,"tid":0,"args":{"name":"python"}}\n' % (ts_meta, gpu_id))
        write(',{"name":"process_labels","ph":"M","ts":%s,"pid":%s,"tid":0,"args":{"labels":"GPU %s"}}\n' % (ts_meta, gpu_id, gpu_id))
        write(',{"name":"process_sort_index","ph":"M","ts":%s,"pid":%s,"tid":0,"args":{"sort_index":%s}}\n' % (ts_meta, gpu_id, 5000000 + gpu_id))

    # GPU stream thread metadata
    for gpu_id, queue_id in sorted(gpu_streams):
        write(',{"name":"thread_name","ph":"M","ts":%s,"pid":%s,"tid":%s,"args":{"name":"stream %s"}}\n' % (ts_meta, gpu_id, queue_id, queue_id))
        write(',{"name":"thread_sort_index","ph":"M","ts":%s,"pid":%s,"tid":%s,"args":{"sort_index":%s}}\n' % (ts_meta, gpu_id, queue_id, queue_id))

    # CPU thread metadata
    for pid, tid in sorted(cpu_tids):
        write(',{"name":"thread_name","ph":"M","ts":%s,"pid":%s,"tid":%s,"args":{"name":"thread %s"}}\n' % (ts_meta, pid, tid, tid))
        write(',{"name":"thread_sort_index","ph":"M","ts":%s,"pid":%s,"tid":%s,"args":{"sort_index":%s}}\n' % (ts_meta, pid, tid, tid))

    # CPU events (cpu_op + user_annotation) and cuda_runtime events
    # Use domain/category from v3 schema, fall back to name matching for v2
    if v3:
        # Build string id lookup for domain/category
        string_lookup = {}
        for row in connection.execute("SELECT id, string FROM rocpd_string"):
            string_lookup[row[0]] = row[1]

        # Build ustring id lookup for args
        ustring_lookup = {}
        for row in connection.execute("SELECT id, string FROM rocpd_ustring WHERE string LIKE '{%'"):
            ustring_lookup[row[0]] = row[1]

        # Single pass over all API events
        kernelapi_data = {}
        if has_kernelapi:
            for row in connection.execute(
                "SELECT api_ptr_id, gridX, gridY, gridZ, "
                "workgroupX, workgroupY, workgroupZ, groupSegmentSize "
                "FROM rocpd_kernelapi"
            ):
                kernelapi_data[row[0]] = row[1:]

        api_ops_by_api = {}
        for row in connection.execute("SELECT id, api_id FROM rocpd_api_ops"):
            api_ops_by_api[row[1]] = row[0]

        # Track forward ops with seq for fwdbwd flows
        fwd_seq_events = {}
        bwd_seq_events = []

        query = (
            "SELECT rocpd_api.id, A.string AS apiName, pid, tid, "
            "rocpd_api.start/1000.0, (rocpd_api.end - rocpd_api.start)/1000.0, "
            "domain_id, category_id, args_id "
            "FROM rocpd_api "
            "INNER JOIN rocpd_string A ON A.id = rocpd_api.apiName_id"
            + range_api
            + " ORDER BY rocpd_api.id"
        )
        for row in connection.execute(query, params_api):
            api_id = row[0]
            api_name = row[1]
            domain = string_lookup.get(row[6], '')
            category = string_lookup.get(row[7], '')
            kineto_cat = _KINETO_DOMAIN_CAT.get((domain, category))
            if kineto_cat is None:
                continue

            pid, tid, ts, dur = row[2], row[3], row[4], row[5]

            if kineto_cat == 'cuda_runtime':
                args_parts = ['"External id":%s' % api_id]
                corr = api_ops_by_api.get(api_id)
                if corr is not None:
                    args_parts.append('"correlation":%s' % corr)
                kinfo = kernelapi_data.get(api_id)
                if kinfo is not None:
                    args_parts.append('"grid":[%s,%s,%s]' % (kinfo[0], kinfo[1], kinfo[2]))
                    args_parts.append('"block":[%s,%s,%s]' % (kinfo[3], kinfo[4], kinfo[5]))
                    args_parts.append('"shared memory":%s' % kinfo[6])
                write(',{"ph":"X","cat":"cuda_runtime","name":%s,"pid":%s,"tid":%s,"ts":%s,"dur":%s,"args":{%s}}\n' % (_s(api_name), pid, tid, ts, dur, ','.join(args_parts)))
            else:
                # Parse args JSON for seq/op_id if available
                args_str = ustring_lookup.get(row[8], '')
                seq = -1
                if args_str and args_str.startswith('{'):
                    try:
                        parsed = json.loads(args_str)
                        seq = parsed.get('seq', -1)
                    except (json.JSONDecodeError, ValueError):
                        pass

                write(',{"ph":"X","cat":"%s","name":%s,"pid":%s,"tid":%s,"ts":%s,"dur":%s,"args":{"External id":%s}}\n' % (kineto_cat, _s(api_name), pid, tid, ts, dur, api_id))

                if seq >= 0:
                    if category == 'function':
                        fwd_seq_events[seq] = (ts, pid, tid)
                    elif category == 'backward_function':
                        bwd_seq_events.append((seq, ts, pid, tid))

    else:
        # v2 fallback: no domain/category columns, use name matching
        fwd_seq_events = {}
        bwd_seq_events = []

        query = (
            "SELECT rocpd_api.id, A.string AS apiName, pid, tid, "
            "rocpd_api.start/1000.0, (rocpd_api.end - rocpd_api.start)/1000.0 "
            "FROM rocpd_api "
            "INNER JOIN rocpd_string A ON A.id = rocpd_api.apiName_id"
            + range_api
            + " ORDER BY rocpd_api.id"
        )
        for row in connection.execute(query, params_api):
            api_name = row[1]
            if api_name.startswith('hip'):
                cat = 'cuda_runtime'
            elif api_name.startswith('iteration') or api_name.startswith('Optimizer.'):
                cat = 'user_annotation'
            elif (api_name.startswith('rpd') or api_name.endswith('::writeRows')
                  or api_name == 'hcc_activity_callback'):
                continue
            else:
                cat = 'cpu_op'

            if cat == 'cuda_runtime':
                write(',{"ph":"X","cat":"cuda_runtime","name":%s,"pid":%s,"tid":%s,"ts":%s,"dur":%s,"args":{"External id":%s}}\n' % (_s(api_name), row[2], row[3], row[4], row[5], row[0]))
            else:
                write(',{"ph":"X","cat":"%s","name":%s,"pid":%s,"tid":%s,"ts":%s,"dur":%s,"args":{"External id":%s}}\n' % (cat, _s(api_name), row[2], row[3], row[4], row[5], row[0]))

    # GPU kernel events
    query = (
        "SELECT B.string AS description, gpuId, queueId, "
        "rocpd_op.start/1000.0, (rocpd_op.end - rocpd_op.start)/1000.0, "
        "rocpd_api_ops.id, rocpd_api.id "
        "FROM rocpd_op "
        "INNER JOIN rocpd_string B ON B.id = rocpd_op.description_id "
        "INNER JOIN rocpd_api_ops ON rocpd_api_ops.op_id = rocpd_op.id "
        "INNER JOIN rocpd_api ON rocpd_api_ops.api_id = rocpd_api.id"
        + range_op
    )
    for row in connection.execute(query, params_op):
        write(',{"ph":"X","cat":"kernel","name":%s,"pid":%s,"tid":%s,"ts":%s,"dur":%s,"args":{"External id":%s,"device":%s,"stream":%s,"correlation":%s}}\n' % (_s(row[0]), row[1], row[2], row[3], row[4], row[6], row[1], row[2], row[5]))

    # gpu_user_annotation: project user_scope events onto GPU timeline
    if v3:
        scopes = connection.execute(
            "SELECT rocpd_api.id, A.string, rocpd_api.start, rocpd_api.end "
            "FROM rocpd_api "
            "INNER JOIN rocpd_string B ON B.id = rocpd_api.category_id "
            "AND B.string = 'user_scope' "
            "INNER JOIN rocpd_string A ON A.id = rocpd_api.apiName_id"
        ).fetchall()
        if scopes:
            ops_by_gpu = _load_ops_by_gpu(connection, range_op, params_op)
            for api_id, api_name, gpu_id, queue_id, gpu_start, gpu_dur in _gpu_user_annotations(ops_by_gpu, scopes):
                write(',{"ph":"X","cat":"gpu_user_annotation","name":%s,"pid":%s,"tid":%s,"ts":%s,"dur":%s,"args":{"External id":%s}}\n' % (_s(api_name), gpu_id, queue_id, gpu_start, gpu_dur, api_id))

    # ac2g flow events (cpu runtime -> gpu kernel)
    query = (
        "SELECT rocpd_api_ops.id, pid, tid, gpuId, queueId, "
        "rocpd_api.start/1000.0, rocpd_op.start/1000.0 "
        "FROM rocpd_api_ops "
        "INNER JOIN rocpd_api ON rocpd_api_ops.api_id = rocpd_api.id "
        "INNER JOIN rocpd_op ON rocpd_api_ops.op_id = rocpd_op.id"
        + range_api
    )
    for row in connection.execute(query, params_api):
        corr = row[0]
        write(',{"ph":"s","id":%s,"pid":%s,"tid":%s,"ts":%s,"cat":"ac2g","name":"ac2g"}\n' % (corr, row[1], row[2], row[5]))
        write(',{"ph":"f","id":%s,"pid":%s,"tid":%s,"ts":%s,"cat":"ac2g","name":"ac2g","bp":"e"}\n' % (corr, row[3], row[4], row[6]))

    # fwdbwd flow events (forward op -> backward op, linked by seq number)
    flow_id = 0
    for seq, bwd_ts, bwd_pid, bwd_tid in bwd_seq_events:
        if seq in fwd_seq_events:
            fwd_ts, fwd_pid, fwd_tid = fwd_seq_events[seq]
            flow_id += 1
            write(',{"ph":"s","id":%s,"pid":%s,"tid":%s,"ts":%s,"cat":"fwdbwd","name":"fwdbwd"}\n' % (flow_id, fwd_pid, fwd_tid, fwd_ts))
            write(',{"ph":"f","id":%s,"pid":%s,"tid":%s,"ts":%s,"cat":"fwdbwd","name":"fwdbwd","bp":"e"}\n' % (flow_id, bwd_pid, bwd_tid, bwd_ts))

    write(']}\n')


def main():
    parser = argparse.ArgumentParser(
        description='Convert RPD to JSON for Chrome Tracing'
    )
    parser.add_argument('input_rpd', type=str, help="input rpd db")
    parser.add_argument('output_json', type=str, nargs='?',
                        help="chrome tracing json output")
    parser.add_argument('--start', type=str,
                        help="start time - us or percentage %%")
    parser.add_argument('--end', type=str,
                        help="end time - us or percentage %%")
    parser.add_argument('--format', type=str, default="object",
                        choices=["object", "array"],
                        help="chrome trace format (default: object)")
    parser.add_argument('--style', type=str, default="rpd",
                        choices=["rpd", "kineto"],
                        help="output style (default: rpd)")
    args = parser.parse_args()

    if args.output_json is None:
        args.output_json = str(
            pathlib.PurePath(args.input_rpd).with_suffix(".json")
        )

    connection = sqlite3.connect(args.input_rpd)

    with open(args.output_json, 'w', encoding="utf-8") as outfile:
        if args.style == "kineto":
            generate_kineto_json(connection, outfile,
                                 start=args.start, end=args.end,
                                 trace_format=args.format,
                                 verbose=True)
        else:
            generate_rpd_json(connection, outfile,
                              start=args.start, end=args.end,
                              trace_format=args.format,
                              verbose=True)

    connection.close()


if __name__ == "__main__":
    main()
