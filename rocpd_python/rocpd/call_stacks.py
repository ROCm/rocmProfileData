###########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
###########################################################################

# Create and auxilarly table to express parent/child api calls
#
#
#

import argparse
import sqlite3
from rocpd.importer import RocpdImportData
from rocpd.metadata import Metadata

from collections import deque

def generateCallStacks(imp):
    meta = Metadata(imp)
    if meta.get("Callstack::Generated") != None:
        raise Exception("Callstack data has already been generated")

    count = 0
    call_inserts = []

    class StackFrame:
        def __init__(self, id, start):
            self.id = id
            self.start = start
            self.child_cpu_time = 0
            pass

    def commitRecords():
        nonlocal call_inserts
        imp.connection.executemany("insert into ext_callstack(id, parent_id, child_id, depth, cpu_time, gpu_time) values (?,?,?,?,?,?)", call_inserts)
        imp.connection.commit()
        call_inserts = []


    for pidtid in imp.connection.execute("select distinct pid, tid from rocpd_api"):
        stack = deque()
        maxdepth = 0;

        for row in imp.connection.execute("select id, start as ts, '1', '' from rocpd_api where pid=? and tid=? UNION ALL select X.id, X.end as timestamp, '-1', Y.gpu_time from rocpd_api X LEFT JOIN (select A.api_id, sum(B.end - B.start) as gpu_time from rocpd_api_ops A join rocpd_op B on B.id = A.op_id group by A.api_id) Y on Y.api_id = X.id where pid=? and tid=? order by ts", (pidtid[0], pidtid[1], pidtid[0], pidtid[1])):
            if row[2] == '1':
                stack.append(StackFrame(row[0], row[1]))
            elif row[2]  == '-1':
                if len(stack) > maxdepth:
                    maxdepth = len(stack)
                depth = len(stack) 
                cpu_time = row[1] - stack[depth - 1].start - stack[depth - 1].child_cpu_time  # cpu duration of returning call
                gpu_time = 0 if row[3] == None else row[3]
                for span in stack:
                    depth = depth - 1
                    if (depth > 0):
                        stack[depth].child_cpu_time = stack[depth].child_cpu_time + cpu_time
                    call_inserts.append((count, span.id, row[0], depth, cpu_time, gpu_time))
                    count = count + 1
                    if (count % 100000 == 99999):
                        commitRecords()
                stack.pop()

        print(f"pid {pidtid[0]}  tid {pidtid[1]}  maxDepth {maxdepth}")

    meta.set("Callstack::Generated", "True")
    commitRecords()


def createCallStackTable(imp):
    meta = Metadata(imp)
    if meta.get("Callstack::Table") != None:
        raise Exception("Callstack table has already been created")

    #Create table
    imp.connection.execute('CREATE TABLE IF NOT EXISTS "ext_callstack" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "parent_id" integer NOT NULL REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "child_id" integer NOT NULL REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "depth" integer NOT NULL, "cpu_time" integer NOT NULL DEFAULT 0, "gpu_time" integer NOT NULL DEFAULT 0)')

    # Make some working views
    imp.connection.execute('CREATE VIEW IF NOT EXISTS callStack_inclusive as select parent_id, sum(cpu_time) as cpu_time, sum(gpu_time) as gpu_time from ext_callstack group by parent_id')
    imp.connection.execute('CREATE VIEW IF NOT EXISTS callStack_exclusive as select parent_id, sum(cpu_time) as cpu_time, sum(gpu_time) as gpu_time from ext_callstack where depth = 0 group by parent_id')
    imp.connection.execute('CREATE VIEW IF NOT EXISTS callStack_inclusive_name as select A.parent_id, B.apiName, B.args, sum(cpu_time) as cpu_time, sum(gpu_time) as gpu_time from ext_callstack A join api B on B.id = A.parent_id group by parent_id')
    imp.connection.execute('CREATE VIEW IF NOT EXISTS callStack_exclusive_name as select A.parent_id, B.apiName, B.args, sum(cpu_time) as cpu_time, sum(gpu_time) as gpu_time from ext_callstack A join api B on B.id = A.parent_id where A.depth = 0 group by parent_id')

    meta.set("Callstack::Table", "True")



if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Generate call stack table to express caller/callee relation')
  parser.add_argument('input_rpd', type=str, help="input rpd db")
  args = parser.parse_args()

  connection = sqlite3.connect(args.input_rpd)

  importData = RocpdImportData()
  importData.resumeExisting(connection) # load the current db state

  createCallStackTable(importData)
  generateCallStacks(importData)

  importData.connection.commit()
