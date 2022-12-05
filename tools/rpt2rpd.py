################################################################################
# Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

#
# Create an rpd file from rocprofiler output files
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

parser = argparse.ArgumentParser(description='convert an RPT-style log to an RPD database')
parser.add_argument('input_rpt', type=str, help="Input file of RPT log entries")
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

#Set up primary keys
string_id = 1
op_id = 1
api_id = 1
hsa_id = 1

# Dicts
strings = {}    # string -> id


#---------------------------------------------
# Everything in one pass
#---------------------------------------------

if args.input_rpt:
    print(f"Importing RPT log from {args.input_rpt}")
    apiExp = re.compile(".*<<hip-api pid:(\d*)\s+tid:(\d*)\.(\d*)\s+\d+\s+\d+\.\d+\s(\w+)\s\'(\w+)\'.*@(\d+).*")
    opExp = re.compile("^profile:  kernel;\s+(\w+);[^;]*;\s+(\d+);\s+(\d+);\s+#(\d+)\.(\d+)\.(\d+);.*")

    count = 0
    api_inserts = [] # rows to bulk insert
    string_inserts = [] # rows to bulk insert
    op_inserts = [] # rows to bulk insert
    api_ops_inserts = [] # rows to bulk insert

    infile = open(args.input_rpt, 'r', encoding="utf-8")

    for line in infile:
        # API calls
        m = apiExp.match(line)
        if m:
            #print(f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)} {m.group(5)} {m.group(6)}")
            try:
                api = strings[m.group(4)]
            except:
                strings[m.group(4)] = string_id
                string_inserts.append((string_id, m.group(4)))
                api = string_id
                string_id = string_id + 1
            try:
                arg = strings[m.group(5)]
            except:
                strings[m.group(5)] = string_id
                string_inserts.append((string_id, m.group(5)))
                arg = string_id
                string_id = string_id + 1

            api_inserts.append((api_id, m.group(1), m.group(2), m.group(6), str(int(m.group(6)) + 10000), api, arg))
            api_id = api_id + 1
            count = count + 1

        # gpu OPs
        m = opExp.match(line)
        if m:
            #print(f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)} {m.group(5)} {m.group(6)}")
            #print(f"kernelNameHere {m.group(2)} {m.group(3)} {m.group(4)} {m.group(5)} {m.group(6)}")
            try:
                name = strings["kernel"]
            except:
                strings["kernel"] = string_id
                string_inserts.append((string_id, "kernel"))
                name = string_id
                string_id = string_id + 1
            try:
                desc = strings[m.group(1)]
            except:
                strings[m.group(1)] = string_id
                string_inserts.append((string_id, m.group(1)))
                desc = string_id
                string_id = string_id + 1

            op_inserts.append((op_id, m.group(4), m.group(5), m.group(6), m.group(2), m.group(3), desc, name))
            op_id = op_id + 1
            count = count + 1
            pass


        if (count % 100000 == 99999):
            connection.executemany("insert into rocpd_string(id, string) values (?,?)", string_inserts)
            connection.executemany("insert into rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?,?)", api_inserts)
            connection.executemany("insert into rocpd_op(id, gpuId, queueId, sequenceId, completionSignal,  start, end, description_id, opType_id) values (?,?,?,?,'',?,?,?,?)", op_inserts)
            connection.executemany("insert into rocpd_api_ops(api_id, op_id) values (?,?)", api_ops_inserts)
            connection.commit()
            api_inserts = []
            op_inserts = []
            string_inserts = []

    connection.executemany("insert into rocpd_string(id, string) values (?,?)", string_inserts)
    connection.executemany("insert into rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?,?)", api_inserts)
    connection.executemany("insert into rocpd_op(id, gpuId, queueId, sequenceId, completionSignal, start, end, description_id, opType_id) values (?,?,?,?,'',?,?,?,?)", op_inserts)
    connection.executemany("insert into rocpd_api_ops(api_id, op_id) values (?,?)", api_ops_inserts)
    connection.commit()
    infile.close()


#Generate Schema Indexes
#connection.execute("")
#connection.execute("")
#connection.execute("")
#connection.execute("")

#Helpful Queries
connection.execute("CREATE VIEW api AS SELECT rocpd_api.id,pid,tid,start,end,A.string AS apiName, B.string AS args FROM rocpd_api INNER JOIN rocpd_string A ON A.id = rocpd_api.apiName_id INNER JOIN rocpd_string B ON B.id = rocpd_api.args_id")
connection.execute("CREATE VIEW op AS SELECT rocpd_op.id,gpuId,queueId,sequenceId,start,end,A.string AS description, B.string AS opType FROM rocpd_op INNER JOIN rocpd_string A ON A.id = rocpd_op.description_id INNER JOIN rocpd_string B ON B.id = rocpd_op.opType_id")
connection.execute("CREATE VIEW top AS SELECT A.string as KernelName, count(A.string) as TotalCalls, sum(rocpd_op.end-rocpd_op.start) / 1000 as TotalDuration, (sum(rocpd_op.end-rocpd_op.start)/count(A.string)) / 1000 as Ave, sum(rocpd_op.end-rocpd_op.start) * 100.0 / (select sum(end-start) from rocpd_op) as Percentage FROM rocpd_op INNER JOIN rocpd_string A ON A.id = rocpd_op.description_id group by KernelName order by TotalDuration desc")
connection.execute("CREATE VIEW busy AS select A.gpuId, GpuTime, WallTime, GpuTime*1.0/WallTime as Busy from (select gpuId, sum(end-start) as GpuTime from rocpd_op group by gpuId) A INNER JOIN (select max(end) - min(start) as WallTime from rocpd_op)")

connection.commit()
connection.close()
