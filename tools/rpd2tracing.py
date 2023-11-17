################################################################################
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
# Format sqlite trace data as json for chrome:tracing
#

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
parser.add_argument('--format', type=str, default="array", help="chome trace format, array or object")
args = parser.parse_args()

print(args)

connection = sqlite3.connect(args.input_rpd)

outfile = open(args.output_json, 'w', encoding="utf-8")

if args.format == "object":
    outfile.write("{\"traceEvents\": ")

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

rangeStringApi = ""
rangeStringOp = ""
if args.start:
    rangeStringApi = "where rocpd_api.start/1000 >= %s"%(args.start)
    rangeStringOp = "where rocpd_op.start/1000 >= %s"%(args.start)
if args.end:
    rangeStringApi = rangeStringApi + " and rocpd_api.start/1000 <= %s"%(args.end) if args.start != None else "where rocpd_api.start/1000 <= %s"%(args.end)
    rangeStringOp = rangeStringOp + " and rocpd_op.start/1000 <= %s"%(args.end) if args.start != None else "where rocpd_op.start/1000 <= %s"%(args.end)

print("Filter: %s"%(rangeStringApi))

# Output Ops
'''
# Hack for busted rocprofiler that can't populate kernel names
for row in connection.execute("select A.string as optype, B.string as description, C.string as arg, gpuId, queueId, rocpd_op.start/1000, (rocpd_op.end-rocpd_op.start) / 1000 from rocpd_op LEFT JOIN rocpd_api_ops on op_id = rocpd_op.id INNER JOIN rocpd_api on api_id = rocpd_api.id INNER JOIN rocpd_string A on A.id = rocpd_op.opType_id INNER Join rocpd_string B on B.id = rocpd_op.description_id INNER JOIN rocpd_string C on C.id = rocpd_api.args_id %s"%(rangeString)):
    try:
        name = row[0] if len(row[2])==0 else row[2]
        outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"name\":\"%s\",\"ts\":\"%s\",\"dur\":\"%s\",\"ph\":\"X\",\"args\":{\"desc\":\"%s\"}}\n"%(row[3], row[4], name, row[5], row[6], row[0]))
    except ValueError:
        outfile.write("")
'''

for row in connection.execute("select A.string as optype, B.string as description, gpuId, queueId, rocpd_op.start/1000, (rocpd_op.end-rocpd_op.start) / 1000 from rocpd_op INNER JOIN rocpd_string A on A.id = rocpd_op.opType_id INNER Join rocpd_string B on B.id = rocpd_op.description_id %s"%(rangeStringOp)):
    try:
        name =  row[0] if len(row[1])==0 else row[1]
        outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"name\":\"%s\",\"ts\":\"%s\",\"dur\":\"%s\",\"ph\":\"X\",\"args\":{\"desc\":\"%s\"}}\n"%(row[2], row[3], name, row[4], row[5], row[0]))
    except ValueError:
        outfile.write("")

#Output apis
for row in connection.execute("select A.string as apiName, B.string as args, pid, tid, rocpd_api.start/1000, (rocpd_api.end-rocpd_api.start) / 1000, (rocpd_api.end != rocpd_api.start) as has_duration from rocpd_api INNER JOIN rocpd_string A on A.id = rocpd_api.apiName_id INNER Join rocpd_string B on B.id = rocpd_api.args_id %s order by rocpd_api.id"%(rangeStringApi)):
    try:
        if row[0]=="UserMarker":
            if row[6] == 0:	# instantanuous "mark" messages
                outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"name\":\"%s\",\"ts\":\"%s\",\"ph\":\"i\",\"s\":\"p\",\"args\":{\"desc\":\"%s\"}}\n"%(row[2], row[3], row[1].replace('"',''), row[4], row[1].replace('"','')))
            else:
                outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"name\":\"%s\",\"ts\":\"%s\",\"dur\":\"%s\",\"ph\":\"X\",\"args\":{\"desc\":\"%s\"}}\n"%(row[2], row[3], row[1].replace('"',''), row[4], row[5], row[1].replace('"','')))
        else:
          outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"name\":\"%s\",\"ts\":\"%s\",\"dur\":\"%s\",\"ph\":\"X\",\"args\":{\"desc\":\"%s\"}}\n"%(row[2], row[3], row[0], row[4], row[5], row[1].replace('"','').replace('\t','')))
    except ValueError:
        outfile.write("")

#Output api->op linkage
for row in connection.execute("select rocpd_api_ops.id, pid, tid, gpuId, queueId, rocpd_api.end/1000 - 2, rocpd_op.start/1000 from rocpd_api_ops INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id %s"%(rangeStringApi)):
    try:
        fromtime = row[5] if row[5] < row[6] else row[6]
        outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"cat\":\"api_op\",\"name\":\"api_op\",\"ts\":\"%s\",\"id\":\"%s\",\"ph\":\"s\"}\n"%(row[1], row[2], fromtime, row[0]))
        outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"cat\":\"api_op\",\"name\":\"api_op\",\"ts\":\"%s\",\"id\":\"%s\",\"ph\":\"f\", \"bp\":\"e\"}\n"%(row[3], row[4], row[6], row[0]))
    except ValueError:
        outfile.write("")

try:
    for row in connection.execute("select A.string as apiName, B.string as args, pid, tid, rocpd_hsaApi.start/1000, (rocpd_hsaApi.end-rocpd_hsaApi.start) / 1000 from rocpd_hsaApi INNER JOIN rocpd_string A on A.id = rocpd_hsaApi.apiName_id INNER Join rocpd_string B on B.id = rocpd_hsaApi.args_id %s order by rocpd_hsaApi.id"%(rangeStringApi)):
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
    for row in connection.execute("select * from (select rocpd_api.start/1000 as ts, \"1\" from rocpd_api_ops INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id AND rocpd_op.gpuId = %s %s UNION ALL select rocpd_op.end/1000, \"-1\" from rocpd_api_ops INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id AND rocpd_op.gpuId = %s %s) order by ts"%(gpuId, rangeStringOp, gpuId, rangeStringOp)):
        try:
           if idle and int(row[1]) > 0:
               idle = 0
               outfile.write(',{"pid":"%s","name":"Idle","ph":"C","ts":%s,"args":{"idle":%s}}\n'%(gpuId, row[0], idle))
           if depth == 1 and int(row[1]) < 0:
               idle = 1
               outfile.write(',{"pid":"%s","name":"Idle","ph":"C","ts":%s,"args":{"idle":%s}}\n'%(gpuId, row[0], idle))
           depth = depth + int(row[1])
           outfile.write(',{"pid":"%s","name":"QueueDepth","ph":"C","ts":%s,"args":{"depth":%s}}\n'%(gpuId, row[0], depth))
        except ValueError:
            outfile.write("")
    if T_end > 0:
            outfile.write(',{"pid":"%s","name":"Idle","ph":"C","ts":%s,"args":{"idle":%s}}\n'%(gpuId, T_end, idle))
            outfile.write(',{"pid":"%s","name":"QueueDepth","ph":"C","ts":%s,"args":{"depth":%s}}\n'%(gpuId, T_end, depth))

# Create SMI counters
try:
    for row in connection.execute("select deviceId, monitorType, start/1000, value from rocpd_monitor"):
        outfile.write(',{"pid":"%s","name":"%s","ph":"C","ts":%s,"args":{"%s":%s}}\n'%(row[0], "", row[2], row[1], row[3]))
    # Output the endpoints of the last range
    for row in connection.execute("select distinct deviceId, monitorType, max(end)/1000, value from rocpd_monitor group by deviceId, monitorType"):
        outfile.write(',{"pid":"%s","name":"%s","ph":"C","ts":%s,"args":{"%s":%s}}\n'%(row[0], "", row[2], row[1], row[3]))
except:
    print("Did not find SMI data")

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
                try:    # Sometimes free addresses are not valid or listed
                    size = sizes[m.group(1)]
                    sizes[m.group(1)] = 0
                    totalSize = totalSize - size;
                    outfile.write(',{"pid":"0","name":"Allocated Memory","ph":"C","ts":%s,"args":{"depth":%s}}\n'%(row[0],totalSize))
                except KeyError:
                    pass
    except ValueError:
        outfile.write("")
if T_end > 0:
    outfile.write(',{"pid":"0","name":"Allocated Memory","ph":"C","ts":%s,"args":{"depth":%s}}\n'%(T_end,totalSize))

#Create "faux calling stack frame" on gpu ops traceS
stacks = {}          # Call stacks built from UserMarker entres.     Key is 'pid,tid'
currentFrame = {}    #"Current GPU frame" (id, name, start, end).    Key is 'pid,tid'

class GpuFrame:
    def __init__(self):
        self.id = 0
        self.name = ''
        self.start = 0
        self.end = 0
        self.gpus = []
        self.totalOps = 0

# FIXME: include 'start' (in ns) so we can ORDER BY it and break ties?
for row in connection.execute("SELECT '0', start/1000, pid, tid, B.string as label, '','','', '' from rocpd_api INNER JOIN rocpd_string A on A.id = rocpd_api.apiName_id AND A.string = 'UserMarker' INNER JOIN rocpd_string B on B.id = rocpd_api.args_id AND rocpd_api.start/1000 != rocpd_api.end/1000 UNION ALL SELECT '1', end/1000, pid, tid, B.string as label, '','','', '' from rocpd_api INNER JOIN rocpd_string A on A.id = rocpd_api.apiName_id AND A.string = 'UserMarker' INNER JOIN rocpd_string B on B.id = rocpd_api.args_id AND rocpd_api.start/1000 != rocpd_api.end/1000 UNION ALL SELECT '2', rocpd_api.start/1000, pid, tid, '' as label, gpuId, queueId, rocpd_op.start/1000, rocpd_op.end/1000 from rocpd_api_ops INNER JOIN rocpd_api ON rocpd_api_ops.api_id = rocpd_api.id INNER JOIN rocpd_op ON rocpd_api_ops.op_id = rocpd_op.id ORDER BY start/1000 asc"):
    try:
        key = (row[2], row[3])    # Key is 'pid,tid'
        if row[0] == '0':  # Frame start
            if key not in stacks:
                stacks[key] = []
            stack = stacks[key].append((row[1], row[4]))
            #print(f"0: new api frame: pid_tid={key} -> stack={stacks}")

        elif row[0] == '1':  #Frame end
            completed = stacks[key].pop()
            #print(f"1: end api frame: pid_tid={key} -> stack={stacks}")

        elif row[0] == '2':  # API + Op
            if key in stacks and len(stacks[key]) > 0:
                frame = stacks[key][-1]
                #print(f"2: Op on {frame} ({len(stacks[key])})")
                gpuFrame = None
                if key not in currentFrame:    # First op under the current api frame
                    gpuFrame = GpuFrame()
                    gpuFrame.id = frame[0]
                    gpuFrame.name = frame[1]
                    gpuFrame.start = row[7]
                    gpuFrame.end = row[8]
                    gpuFrame.gpus.append((row[5], row[6]))
                    gpuFrame.totalOps = 1
                    #print(f"2a: new frame: {gpuFrame.gpus} {gpuFrame.start} {gpuFrame.end} {gpuFrame.end - gpuFrame.start}")
                else:
                    gpuFrame = currentFrame[key]
                    # Another op under the same frame -> union them (but only if they are butt together)
                    if gpuFrame.id == frame[0] and gpuFrame.name == frame[1] and (abs(row[7] - gpuFrame.end) < 200 or abs(gpuFrame.start - row[8]) < 200): 
                    #if gpuFrame.id == frame[0] and gpuFrame.name == frame[1]:    # Another op under the same frame -> union them
                    #if False:   # Turn off frame joining
                        if row[7] < gpuFrame.start: gpuFrame.start = row[7]
                        if row[8] > gpuFrame.end: gpuFrame.end = row[8] 
                        if (row[5], row[6]) not in gpuFrame.gpus: gpuFrame.gpus.append((row[5], row[6]))
                        gpuFrame.totalOps = gpuFrame.totalOps + 1
                        #print(f"2c: union frame: {gpuFrame.gpus} {gpuFrame.start} {gpuFrame.end} {gpuFrame.end - gpuFrame.start}")

                    else:    #This is a new frame - dump the last and make new
                        gpuFrame = currentFrame[key]
                        for dest in gpuFrame.gpus:
                            #print(f"2: OUTPUT: dest={dest} time={gpuFrame.start} -> {gpuFrame.end} Duration={gpuFrame.end - gpuFrame.start} TotalOps={gpuFrame.totalOps}")
                            outfile.write(',{"pid":"%s","tid":"%s","name":"%s","ts":"%s","dur":"%s","ph":"X","args":{"desc":"%s"}}\n'%(dest[0], dest[1], gpuFrame.name, gpuFrame.start - 1, gpuFrame.end - gpuFrame.start + 1, f"UserMarker frame: {gpuFrame.totalOps} ops"))
                        currentFrame.pop(key)

                        # make the first op under the new frame
                        gpuFrame = GpuFrame()
                        gpuFrame.id = frame[0]
                        gpuFrame.name = frame[1]
                        gpuFrame.start = row[7]
                        gpuFrame.end = row[8]
                        gpuFrame.gpus.append((row[5], row[6]))
                        gpuFrame.totalOps = 1
                        #print(f"2b: new frame: {gpuFrame.gpus} {gpuFrame.start} {gpuFrame.end} {gpuFrame.end - gpuFrame.start}")

                currentFrame[key] = gpuFrame

    except ValueError:
        outfile.write("")

outfile.write("]\n")

if args.format == "object":
    outfile.write("} \n")

outfile.close()
connection.close()
