###########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
###########################################################################

#
# Create an rpd file from an nvvp file
#

import sys
import os
import csv
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
import argparse
from os import path

parser = argparse.ArgumentParser(description='convert nvvp database to an RPD database')
parser.add_argument('--cupti_input_file', type=str, help="cupti_runtime_cbid.h with api names")
parser.add_argument('input_nvvp', type=str, help="nvvp input")
parser.add_argument('output_rpd', type=str, help="new output file")
args = parser.parse_args()

if path.exists(args.output_rpd):
    raise Exception(f"Output file: {args.output_rpd} already exists")

connection = sqlite3.connect(args.output_rpd)

#Generate Schema Tables

connection.execute('CREATE TABLE IF NOT EXISTS "rocpd_string" ("id" integer NOT NULL PRIMARY KEY, "string" varchar(4096) NOT NULL)')
connection.execute('CREATE TABLE IF NOT EXISTS "rocpd_op" ("id" integer NOT NULL PRIMARY KEY, "gpuId" integer NOT NULL, "queueId" integer NOT NULL, "sequenceId" integer NOT NULL, "completionSignal" varchar(18) NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "description_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "opType_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED)')
connection.execute('CREATE TABLE IF NOT EXISTS "rocpd_api" ("id" integer NOT NULL PRIMARY KEY, "pid" integer NOT NULL, "tid" integer NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "apiName_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "args_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED)')
connection.execute('CREATE TABLE IF NOT EXISTS "rocpd_api_ops" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "api_id" integer NOT NULL REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "op_id" integer NOT NULL REFERENCES "rocpd_op" ("id") DEFERRABLE INITIALLY DEFERRED)')
connection.execute('CREATE TABLE IF NOT EXISTS "rocpd_hsaApi" ("id" integer NOT NULL PRIMARY KEY, "pid" integer NOT NULL, "tid" integer NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "apiName_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "args_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "return" integer NOT NULL)')

connection.execute('ATTACH DATABASE ? AS nvvp', (args.input_nvvp,))

#for row in connection.execute('SELECT * from nvvp.StringTable'):
#    print(f"{row[0]}   {row[1]}")

# Parse cupti_runtime_cbid.h for API names
# Create a temp cbid table and string entries

#FIXME: DEFERRABLE isn't helping with cbid_id being initially NULL - executemany() does a commit?
connection.execute('CREATE TABLE IF NOT EXISTS "cbid" ("id" integer NOT NULL PRIMARY KEY, "cbid_id" integer REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "string" varchar(4096) NOT NULL)')

infile = open(args.cupti_input_file, 'r', encoding="utf-8")
exp = re.compile("^\s*CUPTI_RUNTIME_TRACE_CBID_(\w+)\s+=\s*([0-9a-fxX]+).*$")
cupti_inserts = [] # rows to bulk insert
for line in infile:
    m = exp.match(line)
    if m:
        cupti_inserts.append((int(m.group(2), 0), m.group(1)))
        #print(f"{m.group(2)} {m.group(1)}")
connection.executemany("insert into cbid(id, string) values (?,?)", cupti_inserts)
infile.close()


# Transfer string table verbatim 
connection.execute("INSERT INTO rocpd_string(id, string) SELECT _id_, value FROM nvvp.StringTable")

# Append API names to string table
connection.execute("INSERT INTO rocpd_string(string) SELECT string from cbid")

# Update foreign keys on cbid
connection.execute("UPDATE cbid SET cbid_id = (SELECT id FROM rocpd_string WHERE rocpd_string.string = cbid.string LIMIT 1)")
#connection.commit()

# Transfer apis from CUPTI_ACTIVITY_KIND_RUNTIME

connection.execute("INSERT INTO rocpd_api (id, pid, tid, start, end, apiName_id, args_id) SELECT _id_, processId, threadId, start, end, C.id, '0' from nvvp.CUPTI_ACTIVITY_KIND_RUNTIME A INNER JOIN cbid B ON B.id = A.cbid INNER JOIN rocpd_string C ON C.id = B.cbid_id")

# Transfer ops from CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
#connection.execute("INSERT INTO rocpd_op (id, gpuId, queueId, sequenceId, completionSignal, start, end, description_id, opType_id) SELECT _id_, deviceId, contextId, streamId, 0, start, end, B.id, B.id from nvvp.CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL A INNER JOIN rocpd_string B on B.id = A.name")
connection.execute("INSERT INTO rocpd_op (id, gpuId, queueId, sequenceId, completionSignal, start, end, description_id, opType_id) SELECT A._id_, deviceId, contextId, streamId, 0, A.start, A.end, D.id, C.cbid_id from nvvp.CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL A INNER JOIN nvvp.CUPTI_ACTIVITY_KIND_RUNTIME B ON A.correlationId = B.correlationId INNER JOIN cbid C on C.id = b.cbid INNER JOIN rocpd_string D ON D.id = A.name")

# Build the api->op bridge table
connection.execute("INSERT INTO rocpd_api_ops(id, api_id, op_id) SELECT A.correlationId, A._id_, B._id_ from nvvp.CUPTI_ACTIVITY_KIND_RUNTIME A INNER JOIN nvvp.CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL B on A.correlationId = B.correlationId")


# Transfer CUPTI_ACTIVITY_KIND_MEMCPY
# Transfer CUPTI_ACTIVITY_KIND_MEMSET
# Transfer CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
# Transfer CUPTI_ACTIVITY_KIND_CUDA_EVENT
# Transfer CUPTI_ACTIVITY_KIND_DRIVER

connection.commit()
#Set up primary keys
string_id = 1
op_id = 1
api_id = 1
hsa_id = 1

# Dicts
strings = {}    # string -> id

connection.commit()

#Generate Schema Indexes
#connection.execute("")
#connection.execute("")
#connection.execute("")
#connection.execute("")

#Helpful Queries
connection.execute("CREATE VIEW api AS SELECT rocpd_api.id,pid,tid,start,end,A.string AS apiName, B.string AS args FROM rocpd_api INNER JOIN rocpd_string A ON A.id = rocpd_api.apiName_id INNER JOIN rocpd_string B ON B.id = rocpd_api.args_id")
connection.execute("CREATE VIEW op AS SELECT rocpd_op.id,gpuId,queueId,sequenceId,start,end,A.string AS description, B.string AS opType FROM rocpd_op INNER JOIN rocpd_string A ON A.id = rocpd_op.description_id INNER JOIN rocpd_string B ON B.id = rocpd_op.opType_id")
connection.execute("CREATE VIEW top AS SELECT A.string as KernelName, count(A.string) as TotalCalls, sum(rocpd_op.end-rocpd_op.start) / 1000 as TotalDuration, (sum(rocpd_op.end-rocpd_op.start)/count(A.string)) / 1000 as Ave, sum(rocpd_op.end-rocpd_op.start) * 100.0 / (select sum(end-start) from rocpd_op) as Percentage FROM rocpd_api_ops INNER JOIN rocpd_op ON rocpd_api_ops.op_id = rocpd_op.id INNER JOIN rocpd_string A ON A.id = rocpd_op.description_id group by KernelName order by TotalDuration desc")
connection.execute("CREATE VIEW busy AS select A.gpuId, GpuTime, WallTime, GpuTime*1.0/WallTime as Busy from (select gpuId, sum(end-start) as GpuTime from rocpd_op group by gpuId) A INNER JOIN (select max(end) - min(start) as WallTime from rocpd_op)")


connection.commit()
connection.close()
