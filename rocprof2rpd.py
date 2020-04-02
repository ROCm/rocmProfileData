# Create an rpd file from rocprofiler output files

import sys
import os
import csv
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
import argparse
from os import path

parser = argparse.ArgumentParser(description='convert rocprofiler output to an RPD database')
parser.add_argument('--ops_input_file', type=str, help="hcc_ops_trace.txt from rocprofiler")
parser.add_argument('--api_input_file', type=str, help="hip_api_trace.txt from rocprofiler")
parser.add_argument('--hsa_input_file', type=str, help="hsa_api_trace.txt from rocprofiler")
parser.add_argument('--roctx_input_file', type=str, help="roctx_trace.txt from rocprofiler")
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
# Api
#---------------------------------------------
if args.api_input_file:
    print(f"Importing hip api calls from {args.api_input_file}")
    exp = re.compile("^(\d*):(\d*)\s+(\d*):(\d*)\s+(\w+)\((.*)\).*$")
    infile = open(args.api_input_file, 'r', encoding="utf-8")
    count = 0
    api_inserts = [] # rows to bulk insert
    string_inserts = [] # rows to bulk insert
    for line in infile:
        m = exp.match(line)
        if m:
            try:
                api = strings[m.group(5)]
            except:
                strings[m.group(5)] = string_id
                string_inserts.append((string_id, m.group(5)))
                api = string_id
                string_id = string_id + 1
            try:
                arg = strings[m.group(6)]
            except:
                strings[m.group(6)] = string_id
                string_inserts.append((string_id, m.group(6)))
                arg = string_id
                string_id = string_id + 1

            api_inserts.append((api_id, m.group(3), m.group(4), m.group(1), m.group(2), api, arg))
            api_id = api_id + 1
            count = count + 1
        if (count % 100000 == 99999):
            connection.executemany("insert into rocpd_string(id, string) values (?,?)", string_inserts)
            connection.executemany("insert into rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?,?)", api_inserts)
            connection.commit()
            api_inserts = []
            string_inserts = []
    connection.executemany("insert into rocpd_string(id, string) values (?,?)", string_inserts)
    connection.executemany("insert into rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?,?)", api_inserts)
    connection.commit()
    infile.close()

#---------------------------------------------
# Ops
#---------------------------------------------
if args.ops_input_file:
    print(f"Importing hcc ops from {args.ops_input_file}")
    exp = re.compile("^(\d*):(\d*)\s+(\d*):(\d*)\s+(\w+):(\d*).*$")
    infile = open(args.ops_input_file, 'r', encoding="utf-8")
    count = 0
    op_inserts = [] # rows to bulk insert
    string_inserts = [] # rows to bulk insert
    api_ops_inserts = [] # rows to bulk insert
    for line in infile:
        m = exp.match(line)
        if m:
            try:
                name = strings[m.group(5)]
                #print(f"   : {m.group(5)} {name}")
            except:
                strings[m.group(5)] = string_id
                string_inserts.append((string_id, m.group(5)))
                #print(f"+++: {m.group(5)} {string_id}")
                name = string_id
                string_id = string_id + 1
            try:
                desc = strings[""]
                #print(f"   : {m.group(6)} {desc}")
            except:
                strings[""] = string_id
                string_inserts.append((string_id, ""))
                #print(f"+++: {m.group(6)} {string_id}")
                desc = string_id
                string_id = string_id + 1

            if len(m.group(6)) > 0:
                api_ops_inserts.append((int(m.group(6)), op_id))

            op_inserts.append((op_id, m.group(3), m.group(4), m.group(1), m.group(2), desc, name))
            op_id = op_id + 1
            count = count + 1
        if (count % 100000 == 99999):
            #print(count+1)
            #print("--------------------------------------------------------------------------")
            #print(string_inserts)
            #print("++++")
            #print(op_inserts)
            #print("####")
            #print(api_ops_inserts)
            connection.executemany("insert into rocpd_string(id, string) values (?,?)", string_inserts)
            connection.executemany("insert into rocpd_op(id, gpuId, queueId, sequenceId, completionSignal,  start, end, description_id, opType_id) values (?,?,?,'','',?,?,?,?)", op_inserts)
            connection.executemany("insert into rocpd_api_ops(api_id, op_id) values (?,?)", api_ops_inserts)
            connection.commit()
            op_inserts = []
            string_inserts = []
            api_ops_inserts = []
    #print("--------------------------------------------------------------------------")
    #print(string_inserts)
    #print("++++")
    #print(op_inserts)
    #print("####")
    #print(api_ops_inserts)
    connection.executemany("insert into rocpd_string(id, string) values (?,?)", string_inserts)
    connection.executemany("insert into rocpd_op(id, gpuId, queueId, sequenceId, completionSignal,  start, end, description_id, opType_id) values (?,?,?,'','',?,?,?,?)", op_inserts)
    connection.executemany("insert into rocpd_api_ops(api_id, op_id) values (?,?)", api_ops_inserts)
    connection.commit()
    infile.close()

#---------------------------------------------
# HSA
#---------------------------------------------
if args.hsa_input_file:
    print(f"Importing hsa api calls from {args.hsa_input_file}")
    exp = re.compile("^(\d*):(\d*)\s+(\d*):(\d*)\s+(\w+)\((.*)\)\s*=\s*(\d*).*$")
    infile = open(args.hsa_input_file, 'r', encoding="utf-8")
    count = 0
    hsa_inserts = [] # rows to bulk insert
    string_inserts = [] # rows to bulk insert
    for line in infile:
        m = exp.match(line)
        if m:
            try:
                api = strings[m.group(5)]
            except:
                strings[m.group(5)] = string_id
                string_inserts.append((string_id, m.group(5)))
                api = string_id
                string_id = string_id + 1
            try:
                arg = strings[m.group(6)]
            except:
                strings[m.group(6)] = string_id
                string_inserts.append((string_id, m.group(6)))
                arg = string_id
                string_id = string_id + 1

            hsa_inserts.append((hsa_id, m.group(3), m.group(4), m.group(1), m.group(2), api, arg, m.group(7)))
            hsa_id = hsa_id + 1
            count = count + 1
        if (count % 100000 == 99999):
            connection.executemany("insert into rocpd_string(id, string) values (?,?)", string_inserts)
            connection.executemany("insert into rocpd_hsaApi(id, pid, tid, start, end, apiName_id, args_id, return) values (?,?,?,?,?,?,?,?)", hsa_inserts)
            connection.commit()
            hsa_inserts = []
            string_inserts = []
    connection.executemany("insert into rocpd_string(id, string) values (?,?)", string_inserts)
    connection.executemany("insert into rocpd_hsaApi(id, pid, tid, start, end, apiName_id, args_id, return) values (?,?,?,?,?,?,?,?)", hsa_inserts)
    connection.commit()
    infile.close()

#---------------------------------------------
# Roctx
#---------------------------------------------

if args.roctx_input_file:
    print(f"Importing markers from {args.roctx_input_file}")
    exp = re.compile("^(\d*)\s+(\d*):(\d*)\s+(\d+):\"(.*)\".*$")
    infile = open(args.roctx_input_file, 'r', encoding="utf-8")
    count = 0
    stack = []
    api_inserts = [] # rows to bulk insert
    string_inserts = [] # rows to bulk insert
    for line in infile:
        m = exp.match(line)
        if m:
            try:
                api = strings["UserMarker"]
            except:
                strings["UserMarker"] = string_id
                string_inserts.append((string_id, "UserMarker"))
                api = string_id
                string_id = string_id + 1
            try:
                arg = strings[m.group(5)]
            except:
                strings[m.group(5)] = string_id
                string_inserts.append((string_id, m.group(5)))
                arg = string_id
                string_id = string_id + 1

            entryType = int(m.group(4))

            if entryType == 0:        # instantaneous marker
                api_inserts.append((api_id, m.group(2), m.group(3), m.group(1), m.group(1), api, arg))
                api_id = api_id + 1
                count = count + 1
            if entryType == 1:
                stack.append((m.group(1), arg))    

            if entryType == 2:
                entry = stack.pop()
                api_inserts.append((api_id, m.group(2), m.group(3), entry[0], m.group(1), api, entry[1]))
                api_id = api_id + 1
                count = count + 1

        if (count % 100000 == 99999):
            connection.executemany("insert into rocpd_string(id, string) values (?,?)", string_inserts)
            connection.executemany("insert into rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?,?)", api_inserts)
            connection.commit()
            api_inserts = []
            string_inserts = []
    connection.executemany("insert into rocpd_string(id, string) values (?,?)", string_inserts)
    connection.executemany("insert into rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?,?)", api_inserts)
    connection.commit()
    infile.close()


# Combine user marker pairs into ranges
api_inserts = []
string_inserts = []
api_removes = []
start_time = None
start_id = None
start_string = None
print(f"Collating user markers")
for row in connection.execute('select rocpd_api.id, B.string, rocpd_api.start, pid, tid, apiName_id from rocpd_api INNER JOIN rocpd_string A on A.id=rocpd_api.apiname_id and A.string="UserMarker" INNER JOIN rocpd_string B on b.id = rocpd_api.args_id where B.string like "%;start;%" or B.string like "%;stop;%" order by B.string, rocpd_api.start'):
    try:
        if (start_time == None): #State machine, state variable STATE 0
            start_id = row[0]
            start_string = row[1]
            start_time = row[2]
        else:                   #STATE 1
            start_toks = start_string.strip('"').split(';')
            stop_toks = row[1].strip('"').split(';')
            if (start_toks[0] != stop_toks[0]):
                print(f"Warning: mismatched user event tags {start_toks[0]} --- {stop_toks[0]}")
            else:
                stop_toks[2] = stop_toks[2].rstrip(')')   # remove stray
                try:
                    api = strings[stop_toks[2]]
                except:
                    strings[stop_toks[2]] = string_id
                    string_inserts.append((string_id, stop_toks[2]))
                    #print(f"{stop_toks[2]}")
                    api = string_id
                    string_id = string_id + 1

                api_inserts.append((api_id, row[3], row[4], start_time, row[2], row[5], api))
                api_id = api_id + 1
                count = count + 1
                api_removes.append((start_id, ))
                api_removes.append((row[0], ))

            start_time = None   # 1 -> 0
    except:
        pass

connection.executemany("insert into rocpd_string(id, string) values (?,?)", string_inserts)
connection.executemany("insert into rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?,?)", api_inserts)
connection.executemany("delete from rocpd_api where id=?", api_removes)

connection.commit()

# Work around an inadequacy if rocprofiler.
# It does not know the name of running kernels.
# Populate kernel names from args of the api that enqueued it

print(f"Fixing missing kernel names")
connection.execute("CREATE TABLE temp.opid_argid('opid' integer NOT NULL PRIMARY KEY, 'argid' integer NOT NULL)")
connection.execute("INSERT INTO opid_argid SELECT rocpd_op.id, rocpd_api.args_id FROM rocpd_api_ops INNER JOIN rocpd_op ON rocpd_op.id = rocpd_api_ops.op_id INNER JOIN rocpd_api ON rocpd_api.id = rocpd_api_ops.api_id")
connection.execute("UPDATE rocpd_op SET description_id = (SELECT argid FROM opid_argid WHERE opid=rocpd_op.id) WHERE rocpd_op.id IN (SELECT opid from opid_argid) AND rocpd_op.opType_id = (SELECT id from rocpd_string where string='hcCommandKernel' limit 1)")
connection.execute("UPDATE rocpd_op SET description_id = (SELECT argid FROM opid_argid WHERE opid=rocpd_op.id) WHERE rocpd_op.id IN (SELECT opid from opid_argid) AND rocpd_op.opType_id = (SELECT id from rocpd_string where string='KernelExecution' limit 1)")
connection.execute("DROP TABLE temp.opid_argid")
connection.commit()

#Helpful Queries
connection.execute("CREATE VIEW api AS SELECT rocpd_api.id,pid,tid,start,end,A.string AS apiName, B.string AS args FROM rocpd_api INNER JOIN rocpd_string A ON A.id = rocpd_api.apiName_id INNER JOIN rocpd_string B ON B.id = rocpd_api.args_id")
connection.execute("CREATE VIEW op AS SELECT rocpd_op.id,gpuId,queueId,sequenceId,start,end,A.string AS description, B.string AS opType FROM rocpd_op INNER JOIN rocpd_string A ON A.id = rocpd_op.description_id INNER JOIN rocpd_string B ON B.id = rocpd_op.opType_id")
connection.execute("CREATE VIEW top AS SELECT A.string as KernelName, count(A.string) as TotalCalls, sum(rocpd_op.end-rocpd_op.start) / 1000 as TotalDuration, (sum(rocpd_op.end-rocpd_op.start)/count(A.string)) / 1000 as Ave, sum(rocpd_op.end-rocpd_op.start) * 100.0 / (select sum(end-start) from rocpd_op) as Percentage FROM rocpd_api_ops INNER JOIN rocpd_op ON rocpd_api_ops.op_id = rocpd_op.id INNER JOIN rocpd_string A ON A.id = rocpd_op.description_id group by KernelName order by TotalDuration desc")
connection.execute("CREATE VIEW busy AS select A.gpuId, GpuTime, WallTime, GpuTime*1.0/WallTime as Busy from (select gpuId, sum(end-start) as GpuTime from rocpd_op group by gpuId) A INNER JOIN (select max(end) - min(start) as WallTime from rocpd_op)")
connection.commit()

#Generate Schema Indexes
#connection.execute("")
#connection.execute("")
#connection.execute("")
#connection.execute("")

connection.close()
