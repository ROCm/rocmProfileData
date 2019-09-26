# Format sqlite trace data as sjon for chrome:tracing

import sys
import os
import csv
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description='convert RPD to json for chrome tracing')
parser.add_argument('input_rpd', type=str, help="input rpd db")
parser.add_argument('output_json', type=str, help="chrome tracing json output")
parser.add_argument('--start', type=int, help="start timestamp")
parser.add_argument('--end', type=int, help="end timestamp")
args = parser.parse_args()

print(args)

connection = sqlite3.connect(args.input_rpd)

outfile = open(args.output_json, 'w', encoding="utf-8")
    
outfile.write("[ {}\n");

for row in connection.execute("select distinct gpuId from rocpd_op"):
    try:
        outfile.write(",{\"name\": \"process_name\", \"ph\": \"M\", \"pid\":\"%s\",\"args\":{\"name\":\"%s\"}}\n"%(row[0], "GPU"+str(row[0])))
        outfile.write(",{\"name\": \"process_sort_index\", \"ph\": \"M\", \"pid\":\"%s\",\"args\":{\"sort_index\":\"%s\"}}\n"%(row[0], row[0] + 1000000))
    except ValueError:
        outfile.write("")

for row in connection.execute("select distinct pid, tid from rocpd_api"):
    try:
        outfile.write(',{"name":"thread_name","ph":"M","pid":"%s","tid":"%s","args":{"name":"%s"}}\n'%(row[0], row[1], "Hip " + str(row[1])))
        outfile.write(',{"name":"thread_sort_index","ph":"M","pid":"%s","tid":"%s","args":{"sort_index":"%s"}}\n'%(row[0], row[1], row[1] * 2))
    except ValueError:
        outfile.write("")

try:
    # FIXME - these aren't rendering correctly in chrome://tracing
    for row in connection.execute("select distinct pid, tid from rocpd_hsaApi"):
        try:
            outfile.write(',{"name":"thread_name","ph":"M","pid":"%s","tid":"%s","args":{"name":"%s"}}\n'%(row[0], row[1], "HSA " + str(row[1])))
            outfile.write(',{"name":"thread_sort_index","ph":"M","pid":"%s","tid":"%s","args":{"sort_index":"%s"}}\n'%(row[0], row[1], row[1] * 2 - 1))
        except ValueError:
            outfile.write("")
except:
    pass

rangeString = ""
if args.start:
    rangeString = "where rocpd_api.start/1000 >= %s"%(args.start)
if args.end:
    rangeString = rangeString + " and rocpd_api.start/1000 <= %s"%(args.end) if args.start != None else "where rocpd_api.start/1000 <= %s"%(args.end)

print("Filter: %s"%(rangeString))

for row in connection.execute("select A.string as optype, B.string as description, C.string as arg, gpuId, queueId, rocpd_op.start/1000, (rocpd_op.end-rocpd_op.start) / 1000 from rocpd_op LEFT JOIN rocpd_api_ops on op_id = rocpd_op.id INNER JOIN rocpd_api on api_id = rocpd_api.id INNER JOIN rocpd_string A on A.id = rocpd_op.opType_id INNER Join rocpd_string B on B.id = rocpd_op.description_id INNER JOIN rocpd_string C on C.id = rocpd_api.args_id %s"%(rangeString)):
    try:
        name = row[0] if len(row[2])==0 else row[2]
        outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"name\":\"%s\",\"ts\":\"%s\",\"dur\":\"%s\",\"ph\":\"X\",\"args\":{\"desc\":\"%s\"}}\n"%(row[3], row[4], name, row[5], row[6], row[0]))
    except ValueError:
        outfile.write("")

for row in connection.execute("select A.string as apiName, B.string as args, pid, tid, rocpd_api.start/1000, (rocpd_api.end-rocpd_api.start) / 1000 from rocpd_api INNER JOIN rocpd_string A on A.id = rocpd_api.apiName_id INNER Join rocpd_string B on B.id = rocpd_api.args_id %s order by rocpd_api.id"%(rangeString)):
    try:
        if row[0]=="USER_EVENT":
            if row[5] == 0:
                outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"name\":\"%s\",\"ts\":\"%s\",\"ph\":\"i\",\"s\":\"p\",\"args\":{\"desc\":\"%s\"}}\n"%(row[2], row[3], row[1].replace('"',''), row[4], row[1].replace('"','')))
            else:
                outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"name\":\"%s\",\"ts\":\"%s\",\"dur\":\"%s\",\"ph\":\"X\",\"args\":{\"desc\":\"%s\"}}\n"%(row[2], row[3], row[1].replace('"',''), row[4], row[5], row[1].replace('"','')))
        else:
          outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"name\":\"%s\",\"ts\":\"%s\",\"dur\":\"%s\",\"ph\":\"X\",\"args\":{\"desc\":\"%s\"}}\n"%(row[2], row[3], row[0], row[4], row[5], row[1].replace('"','')))
    except ValueError:
        outfile.write("")

for row in connection.execute("select rocpd_api_ops.id, pid, tid, gpuId, queueId, rocpd_api.end/1000 - 2, rocpd_op.start/1000 from rocpd_api_ops INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id %s"%(rangeString)):
    try:
        outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"cat\":\"api_op\",\"name\":\"api_op\",\"ts\":\"%s\",\"id\":\"%s\",\"ph\":\"s\"}\n"%(row[1], row[2], row[5], row[0]))
        outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"cat\":\"api_op\",\"name\":\"api_op\",\"ts\":\"%s\",\"id\":\"%s\",\"ph\":\"f\", \"bp\":\"e\"}\n"%(row[3], row[4], row[6], row[0]))
    except ValueError:
        outfile.write("")

try:
    for row in connection.execute("select A.string as apiName, B.string as args, pid, tid, rocpd_hsaApi.start/1000, (rocpd_hsaApi.end-rocpd_hsaApi.start) / 1000 from rocpd_hsaApi INNER JOIN rocpd_string A on A.id = rocpd_hsaApi.apiName_id INNER Join rocpd_string B on B.id = rocpd_hsaApi.args_id %s order by rocpd_hsaApi.id"%(rangeString)):
        try:
            outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"name\":\"%s\",\"ts\":\"%s\",\"dur\":\"%s\",\"ph\":\"X\",\"args\":{\"desc\":\"%s\"}}\n"%(row[2], row[3]+1, row[0], row[4], row[5], row[1].replace('"','')))
        except ValueError:
            outfile.write("")
except:
    pass


#
# Counters
#

# Counters should extend to the last event in the trace.  This means they need to have a value at Tend.
# Figure out when that is

T_end = 0
for row in connection.execute("SELECT max(end)/1000 from (SELECT end from rocpd_api UNION ALL SELECT end from rocpd_op)"):
    T_end = int(row[0])

# Loop over GPU for per-gpu counters
gpuIdsPresent = []
for row in connection.execute("SELECT DISTINCT gpuId FROM rocpd_op"):
    gpuIdsPresent.append(row[0])

for gpuId in gpuIdsPresent:
    print(f"Creating counters for: {gpuId}")

    #Create the queue depth counter
    depth = 0
    idle = 1
    for row in connection.execute("select * from (select rocpd_api.start/1000 as ts, \"1\" from rocpd_api_ops INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id AND rocpd_op.gpuId = %s %s UNION ALL select rocpd_op.end/1000, \"-1\" from rocpd_api_ops INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id AND rocpd_op.gpuId = %s %s) order by ts"%(gpuId, rangeString, gpuId, rangeString)):
        try:
           if idle and int(row[1]) > 0:
               idle = 0
               outfile.write(",{\"pid\":\"0\",\"name\":\"Idle\",\"ph\":\"C\",\"ts\":%s,\"args\":{\"idle\":%s}}\n"%(row[0],idle))
           if depth == 1 and int(row[1]) < 0:
               idle = 1
               outfile.write(",{\"pid\":\"0\",\"name\":\"Idle\",\"ph\":\"C\",\"ts\":%s,\"args\":{\"idle\":%s}}\n"%(row[0],idle))
           depth = depth + int(row[1])
           outfile.write(",{\"pid\":\"0\",\"name\":\"QueueDepth\",\"ph\":\"C\",\"ts\":%s,\"args\":{\"depth\":%s}}\n"%(row[0],depth))
        except ValueError:
            outfile.write("")
if T_end > 0:
            outfile.write(",{\"pid\":\"0\",\"name\":\"Idle\",\"ph\":\"C\",\"ts\":%s,\"args\":{\"idle\":%s}}\n"%(T_end,idle))
            outfile.write(",{\"pid\":\"0\",\"name\":\"QueueDepth\",\"ph\":\"C\",\"ts\":%s,\"args\":{\"depth\":%s}}\n"%(T_end,depth))

#Create the (global) memory counter
sizes = {}    # address -> size
totalSize = 0
exp = re.compile("^ptr\((.*)\)\s+size\((.*)\)$")
exp2 = re.compile("^ptr\((.*)\)$")
for row in connection.execute("SELECT rocpd_api.end/1000 as ts, B.string, '1'  FROM rocpd_api INNER JOIN rocpd_string A ON A.id=rocpd_api.apiName_id INNER JOIN rocpd_string B ON B.id=rocpd_api.args_id WHERE A.string='hipFree' UNION ALL SELECT rocpd_api.start/1000, B.string, '0' FROM rocpd_api INNER JOIN rocpd_string A ON A.id=rocpd_api.apiName_id INNER JOIN rocpd_string B ON B.id=rocpd_api.args_id WHERE A.string='hipMalloc' ORDER BY ts asc"):
    try:
        if row[2] == '0':  #malloc
            m = exp.match(row[1])
            if m:
                size = int(m.group(2), 16)
                totalSize = totalSize + size
                sizes[m.group(1)] = size
                outfile.write(',{"pid":"0","name":"Allocated Memory","ph":"C","ts":%s,"args":{"depth":%s}}\n'%(row[0],totalSize))
        else:              #free
            m = exp2.match(row[1])
            if m:
                size = sizes[m.group(1)]
                sizes[m.group(1)] = 0
                totalSize = totalSize - size;
                outfile.write(',{"pid":"0","name":"Allocated Memory","ph":"C","ts":%s,"args":{"depth":%s}}\n'%(row[0],totalSize))
    except ValueError:
        outfile.write("")
if T_end > 0:
    outfile.write(',{"pid":"0","name":"Allocated Memory","ph":"C","ts":%s,"args":{"depth":%s}}\n'%(T_end,totalSize))

outfile.write("]\n")
outfile.close()
connection.close()
