###########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
###########################################################################

# Create an auxiliary table to express parent/child API calls (N×D expansion).
#
# For each frame F and each of its N ancestor frames A, we emit one row:
#   (parent=A, child=F, depth=D, cpu_time=exclusive(F), gpu_time=exclusive_gpu(F))
#
# Two implementations are provided:
#   generateCallStacksPython  — Python loop with stack replay (default, 15s)
#   generateCallStacksSQL     — Pure SQL with window functions + recursive CTE (16s)
#
# Both produce identical output (verified element-wise on 2.6M rows).

import argparse
import sqlite3
from rocpd.importer import RocpdImportData
from rocpd.metadata import Metadata


# ---------------------------------------------------------------------------
# Python implementation (default) — stack replay per thread
# ---------------------------------------------------------------------------
def generateCallStacksPython(imp):
    """Replay each thread's event stream to build N×D callstack rows.

    For each (pid, tid), we interleave start/end events as an ordered stream.
    On start: push frame onto stack.
    On end:   pop frame, compute its exclusive CPU time, emit one row per
              ancestor in the stack, and propagate the child's time upward.

    The stack is a list of [frame_id, start_ts, child_cpu_time] mutable lists.
    When frame F returns with exclusive time T, we walk the stack bottom-up:
        for each ancestor A at stack[i]:
            if A is not the returning frame: accumulate T into A.child_cpu_time
            emit row (parent=A, child=F, depth=d, cpu_time=T)
    This ensures each ancestor reports only its own exclusive time at depth=0.
    """
    meta = Metadata(imp)
    if meta.get("Callstack::Generated") != None:
        raise Exception("Callstack data has already been generated")

    # Pre-compute GPU time per API (faster than LEFT JOIN in event query)
    gpu_times = {}
    for row in imp.connection.execute(
        "SELECT A.api_id, SUM(B.end - B.start) FROM rocpd_api_ops A "
        "JOIN rocpd_op B ON B.id = A.op_id GROUP BY A.api_id"):
        gpu_times[row[0]] = row[1]

    # Collect all rows in memory, then bulk-insert via single executemany
    call_inserts = []

    for pidtid in imp.connection.execute(
        "SELECT DISTINCT pid, tid FROM rocpd_api ORDER BY pid, tid"):
        # Stack frame: [id, start_ts, child_cpu_time] — mutable list
        stack = []
        maxdepth = 0

        for row in imp.connection.execute(
            "SELECT id, start, 1 FROM rocpd_api WHERE pid=? AND tid=? "
            "UNION ALL "
            "SELECT id, end, -1 FROM rocpd_api WHERE pid=? AND tid=? "
            "ORDER BY 2, 3 DESC",
            (pidtid[0], pidtid[1], pidtid[0], pidtid[1])):
            id, ts, typ = row
            if typ > 0:
                stack.append([id, ts, 0])
            else:
                depth = len(stack)
                if depth > maxdepth:
                    maxdepth = depth
                # Exclusive CPU time = duration minus time spent in children
                cpu_time = ts - stack[depth - 1][1] - stack[depth - 1][2]
                gpu_time = gpu_times.get(id, 0)

                # Walk stack bottom-up: propagate time, emit N×D rows
                d = depth
                for i in range(depth):
                    d -= 1
                    if d > 0:
                        # Add this frame's time to parent's accumulated children
                        stack[d - 1][2] += cpu_time
                    call_inserts.append(
                        (stack[i][0], id, d, cpu_time, gpu_time))
                stack.pop()

        print(f"pid {pidtid[0]}  tid {pidtid[1]}  maxDepth {maxdepth}")

    # Single bulk insert
    if call_inserts:
        imp.connection.executemany(
            "INSERT INTO ext_callstack(parent_id, child_id, depth, cpu_time, gpu_time) "
            "VALUES (?,?,?,?,?)",
            call_inserts)

    meta.set("Callstack::Generated", "True")


# ---------------------------------------------------------------------------
# SQL implementation (alternative) — window functions + recursive CTE
# ---------------------------------------------------------------------------
def generateCallStacksSQL(imp):
    """Pure SQL approach: compute depth via window function, parent via containment,
    exclusive time via aggregation, then expand N×D via a recursive CTE.

    Phase 1 — Depth:
        Build an interleaved event stream (start=+1, end=-1) and compute depth
        as a running sum via SUM() OVER (PARTITION BY pid, tid ORDER BY ts).
        Keep only start events (evt=1) as "opens" — frames with their depth.

    Phase 2 — Slim tree:
        For each frame at depth >= 2, find its parent: the tightest containing
        frame at depth-1 (same pid/tid, start <= this.start, end >= this.end).
        Use a correlated subquery with the redundant WHERE trick:
            AND p.id IN (SELECT id FROM opens)
        This forces SQLite's query planner to use the index on opens
        (without it, the planner does a full table scan and times out).
        Store inclusive cpu_time = end - start.

    Phase 3 — Exclusive time:
        Aggregate inclusive time of direct children per parent.
        Exclusive = inclusive - SUM(children's inclusive).

    Phase 4 — N×D expansion:
        Use a recursive CTE: base case is depth=0 (self-reference for all frames
        with exclusive time). Recursive case walks up the slim_tree edges,
        carrying the leaf frame's exclusive cpu_time and gpu_time upward.
    """
    meta = Metadata(imp)
    if meta.get("Callstack::Generated") != None:
        raise Exception("Callstack data has already been generated")

    conn = imp.connection

    # GPU pre-compute
    conn.execute('CREATE TEMP TABLE api_gpu AS '
                 'SELECT api_id, SUM(end - start) AS gpu_time '
                 'FROM rocpd_api_ops A JOIN rocpd_op B ON B.id = A.op_id '
                 'GROUP BY api_id')
    conn.execute('CREATE INDEX api_gpu_idx ON api_gpu(api_id)')

    # Phase 1: Opens with depth via window function
    # Interleave start (+1) and end (-1) events, compute running sum as depth,
    # filter to start events only (evt=1) to get frame opens.
    conn.execute(
        'CREATE TEMP TABLE opens AS '
        'SELECT id, pid, tid, start, end, depth FROM ('
        '    SELECT id, pid, tid, start, end, evt, '
        '        SUM(evt) OVER ('
        '            PARTITION BY pid, tid '
        '            ORDER BY ts, evt DESC '
        '            ROWS UNBOUNDED PRECEDING) AS depth '
        '    FROM ('
        '        SELECT id, pid, tid, start, end, start AS ts, 1 AS evt '
        '        FROM rocpd_api '
        '        UNION ALL '
        '        SELECT id, pid, tid, start, end, end AS ts, -1 AS evt '
        '        FROM rocpd_api'
        '    )'
        ') WHERE evt = 1')
    conn.execute('CREATE INDEX opens_idx ON opens(pid, tid, depth, start DESC, end)')
    conn.execute('ANALYZE')

    # Phase 2: Slim tree — parent assignment via temporal containment
    # For each frame at depth >= 2, find the tightest containing parent at depth-1.
    # The redundant "AND p.id IN (SELECT id FROM opens)" forces index usage.
    conn.execute(
        'CREATE TEMP TABLE slim_tree AS '
        'SELECT child.id AS child_id, '
        '    (SELECT p.id FROM opens p '
        '     WHERE p.pid = child.pid AND p.tid = child.tid '
        '     AND p.depth = child.depth - 1 '
        '     AND p.start <= child.start AND p.end >= child.end '
        '     AND p.id IN (SELECT id FROM opens) '
        '     ORDER BY p.start DESC LIMIT 1) AS parent_id, '
        '    (child.end - child.start) AS inclusive_cpu, '
        '    COALESCE(ag.gpu_time, 0) AS gpu_time '
        'FROM opens child '
        'LEFT JOIN api_gpu ag ON ag.api_id = child.id '
        'WHERE child.depth > 1')

    # Phase 3: Exclusive time = inclusive - SUM(direct children's inclusive)
    conn.execute(
        'CREATE TEMP TABLE child_agg AS '
        'SELECT parent_id, SUM(inclusive_cpu) AS total_child_inclusive '
        'FROM slim_tree GROUP BY parent_id')
    conn.execute('CREATE INDEX ca_idx ON child_agg(parent_id)')

    conn.execute(
        'CREATE TEMP TABLE slim_exclusive AS '
        'SELECT s.child_id, s.parent_id, '
        '    s.inclusive_cpu - COALESCE(ca.total_child_inclusive, 0) AS cpu_time, '
        '    s.gpu_time '
        'FROM slim_tree s '
        'LEFT JOIN child_agg ca ON s.child_id = ca.parent_id')
    conn.execute('CREATE INDEX slim_ex_child ON slim_exclusive(child_id)')
    conn.execute('CREATE INDEX slim_ex_parent ON slim_exclusive(parent_id)')
    conn.execute('ANALYZE')

    # Phase 4: N×D expansion via recursive CTE
    # Base case: depth=0, self-reference for all frames with exclusive time.
    # Recursive case: walk up slim_exclusive edges, carrying leaf cpu_time/gpu_time.
    conn.execute(
        'INSERT INTO ext_callstack (parent_id, child_id, depth, cpu_time, gpu_time) '
        'WITH RECURSIVE cs(parent_id, child_id, depth, cpu_time, gpu_time) AS ('
        '    SELECT o.id, o.id, 0, '
        '        (o.end - o.start) - COALESCE(ca.total_child_inclusive, 0), '
        '        COALESCE(ag.gpu_time, 0) '
        '    FROM opens o '
        '    LEFT JOIN api_gpu ag ON ag.api_id = o.id '
        '    LEFT JOIN child_agg ca ON o.id = ca.parent_id '
        '    UNION ALL '
        '    SELECT se.parent_id, cs.child_id, cs.depth + 1, cs.cpu_time, cs.gpu_time '
        '    FROM slim_exclusive se JOIN cs ON cs.parent_id = se.child_id '
        ') '
        'SELECT parent_id, child_id, depth, cpu_time, gpu_time FROM cs')

    meta.set("Callstack::Generated", "True")


# ---------------------------------------------------------------------------
# Table setup and views
# ---------------------------------------------------------------------------
def createCallStackTable(imp):
    meta = Metadata(imp)
    if meta.get("Callstack::Table") != None:
        raise Exception("Callstack table has already been created")

    imp.connection.execute('DROP TABLE IF EXISTS "ext_callstack"')
    imp.connection.execute(
        'CREATE TABLE "ext_callstack" ('
        '"id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, '
        '"parent_id" integer NOT NULL, '
        '"child_id" integer NOT NULL, '
        '"depth" integer NOT NULL, '
        '"cpu_time" integer NOT NULL DEFAULT 0, '
        '"gpu_time" integer NOT NULL DEFAULT 0)')

    # Views for downstream aggregation
    imp.connection.execute(
        'CREATE VIEW IF NOT EXISTS callStack_inclusive '
        'AS SELECT parent_id, SUM(cpu_time) AS cpu_time, SUM(gpu_time) AS gpu_time '
        'FROM ext_callstack GROUP BY parent_id')
    imp.connection.execute(
        'CREATE VIEW IF NOT EXISTS callStack_exclusive '
        'AS SELECT parent_id, SUM(cpu_time) AS cpu_time, SUM(gpu_time) AS gpu_time '
        'FROM ext_callstack WHERE depth = 0 GROUP BY parent_id')
    imp.connection.execute(
        'CREATE VIEW IF NOT EXISTS callStack_inclusive_name '
        'AS SELECT A.parent_id, B.apiName, B.args, SUM(A.cpu_time) AS cpu_time, '
        'SUM(A.gpu_time) AS gpu_time '
        'FROM ext_callstack A JOIN api B ON B.id = A.parent_id GROUP BY A.parent_id')
    imp.connection.execute(
        'CREATE VIEW IF NOT EXISTS callStack_exclusive_name '
        'AS SELECT A.parent_id, B.apiName, B.args, SUM(A.cpu_time) AS cpu_time, '
        'SUM(A.gpu_time) AS gpu_time '
        'FROM ext_callstack A JOIN api B ON B.id = A.parent_id '
        'WHERE A.depth = 0 GROUP BY A.parent_id')

    meta.set("Callstack::Table", "True")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate call stack table to express caller/callee relation')
    parser.add_argument('input_rpd', type=str, help="input rpd db")
    parser.add_argument('--sql', action='store_true',
        help='Use pure SQL implementation instead of Python (slightly slower, same result)')
    args = parser.parse_args()

    connection = sqlite3.connect(args.input_rpd, isolation_level=None)
    importData = RocpdImportData()
    importData.resumeExisting(connection)

    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA synchronous=NORMAL')
    connection.execute('PRAGMA cache_size=-64000')

    createCallStackTable(importData)
    connection.execute('BEGIN')

    if args.sql:
        generateCallStacksSQL(importData)
    else:
        generateCallStacksPython(importData)

    # Create indexes after bulk insert (required for downstream queries)
    connection.execute('CREATE INDEX idx_ext_parent ON ext_callstack(parent_id)')
    connection.execute('CREATE INDEX idx_ext_child ON ext_callstack(child_id)')
    connection.execute('CREATE INDEX idx_ext_depth ON ext_callstack(depth)')
    connection.execute('COMMIT')
