#!/usr/bin/env python3

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
# Format sqlite trace data as json for chrome:tracing - Multi-file merger
#

import sys
import os
import csv
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description='convert and merge multiple RPD files to json for chrome tracing')
parser.add_argument('input_rpd', type=str, nargs='+', help="input rpd db files (can specify multiple)")
parser.add_argument('output_json', type=str, nargs='?', help="chrome tracing json output")
parser.add_argument('--start', type=str, help="start time - default us or percentage %%. Number only is interpreted as us. Number with %% is interpreted as percentage")
parser.add_argument('--end', type=str, help="end time - default us or percentage %%. See help for --start")
parser.add_argument('--format', type=str, default="object", help="chrome trace format, array or object")
args = parser.parse_args()

if args.output_json is None:
    import pathlib
    if len(args.input_rpd) == 1:
        args.output_json = pathlib.PurePath(args.input_rpd[0]).with_suffix(".json")
    else:
        args.output_json = "merged_trace.json"

outfile = open(args.output_json, 'w', encoding="utf-8")

if args.format == "object":
    outfile.write("{\"traceEvents\": ")

outfile.write("[ {}\n")

def process_rpd_file(rpd_file, node_id, outfile, args):
    """Process a single RPD file with node prefix"""
    
    print(f"\n{'='*80}")
    print(f"Processing Node {node_id}: {rpd_file}")
    print(f"{'='*80}")
    
    connection = sqlite3.connect(rpd_file)
    
    # Node prefix for display names
    node_prefix = f"Node{node_id}"
    
    # PID offset to avoid conflicts between files
    pid_offset = node_id * 100000
    
    # GPU metadata
    for row in connection.execute("select distinct gpuId from rocpd_op"):
        try:
            gpu_pid = row[0] + pid_offset
            outfile.write(",{\"name\": \"process_name\", \"ph\": \"M\", \"pid\":\"%s\",\"args\":{\"name\":\"%s GPU%s\"}}\n"%(gpu_pid, node_prefix, row[0]))
            outfile.write(",{\"name\": \"process_sort_index\", \"ph\": \"M\", \"pid\":\"%s\",\"args\":{\"sort_index\":\"%s\"}}\n"%(gpu_pid, gpu_pid + 1000000))
        except ValueError:
            pass

    # Thread metadata for HIP APIs
    for row in connection.execute("select distinct pid, tid from rocpd_api"):
        try:
            adj_pid = row[0] + pid_offset
            outfile.write(',{"name":"thread_name","ph":"M","pid":"%s","tid":"%s","args":{"name":"%s Hip %s"}}\n'%(adj_pid, row[1], node_prefix, row[1]))
            outfile.write(',{"name":"thread_sort_index","ph":"M","pid":"%s","tid":"%s","args":{"sort_index":"%s"}}\n'%(adj_pid, row[1], row[1] * 2))
        except ValueError:
            pass

    # Thread metadata for HSA APIs
    try:
        for row in connection.execute("select distinct pid, tid from rocpd_hsaApi"):
            try:
                adj_pid = row[0] + pid_offset
                outfile.write(',{"name":"thread_name","ph":"M","pid":"%s","tid":"%s","args":{"name":"%s HSA %s"}}\n'%(adj_pid, row[1], node_prefix, row[1]))
                outfile.write(',{"name":"thread_sort_index","ph":"M","pid":"%s","tid":"%s","args":{"sort_index":"%s"}}\n'%(adj_pid, row[1], row[1] * 2 - 1))
            except ValueError:
                pass
    except:
        pass

    # Time range calculation
    rangeStringApi = ""
    rangeStringOp = ""
    rangeStringMonitor = ""
    min_time = connection.execute("select MIN(start) from rocpd_api;").fetchall()[0][0]
    max_time = connection.execute("select MAX(end) from rocpd_api;").fetchall()[0][0]
    
    if min_time is None:
        print(f"Warning: Trace file {rpd_file} is empty, skipping...")
        connection.close()
        return

    print(f"Timestamps for {node_prefix}:")
    print(f"\t    first: \t{min_time/1000} us")
    print(f"\t     last: \t{max_time/1000} us")
    print(f"\t duration: \t{(max_time-min_time) / 1000000000} seconds")

    start_time = min_time/1000
    end_time = max_time/1000

    if args.start:
        if "%" in args.start:
            start_time = ( (max_time - min_time) * ( int( args.start.replace("%","") )/100 ) + min_time )/1000
        else:
            start_time = int(args.start)
        rangeStringApi = "where rocpd_api.start/1000 >= %s"%(start_time)
        rangeStringOp = "where rocpd_op.start/1000 >= %s"%(start_time)
        rangeStringMonitor = "where start/1000 >= %s"%(start_time)
    
    if args.end:
        if "%" in args.end:
            end_time = ( (max_time - min_time) * ( int( args.end.replace("%","") )/100 ) + min_time )/1000
        else:
            end_time = int(args.end)

        rangeStringApi = rangeStringApi + " and rocpd_api.start/1000 <= %s"%(end_time) if args.start != None else "where rocpd_api.start/1000 <= %s"%(end_time)
        rangeStringOp = rangeStringOp + " and rocpd_op.start/1000 <= %s"%(end_time) if args.start != None else "where rocpd_op.start/1000 <= %s"%(end_time)
        rangeStringOp = rangeStringOp + " and rocpd_op.start/1000 <= %s"%(end_time) if args.start != None else "where rocpd_op.start/1000 <= %s"%(end_time)
        rangeStringMonitor = rangeStringMonitor + " and start/1000 <= %s"%(end_time) if args.start != None else "where start/1000 <= %s"%(end_time)

    print(f"\nFilter for {node_prefix}: {rangeStringApi}")
    print(f"Output duration: {(end_time-start_time)/1000000} seconds")

    # Output Ops
    for row in connection.execute("select A.string as optype, B.string as description, gpuId, queueId, rocpd_op.start/1000.0, (rocpd_op.end-rocpd_op.start) / 1000.0 from rocpd_op INNER JOIN rocpd_string A on A.id = rocpd_op.opType_id INNER Join rocpd_string B on B.id = rocpd_op.description_id %s"%(rangeStringOp)):
        try:
            name = row[0] if len(row[1])==0 else row[1]
            gpu_pid = row[2] + pid_offset
            outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"name\":\"%s\",\"ts\":\"%s\",\"dur\":\"%s\",\"ph\":\"X\",\"args\":{\"desc\":\"%s\"}}\n"%(gpu_pid, row[3], name, row[4], row[5], row[0]))
        except ValueError:
            pass

    # Output Graph executions on GPU
    try:
        for row in connection.execute('select graphExec, gpuId, queueId, min(start)/1000.0, (max(end)-min(start))/1000.0, count(*) from rocpd_graphLaunchapi A join rocpd_api_ops B on B.api_id = A.api_ptr_id join rocpd_op C on C.id = B.op_id %s group by api_ptr_id'%(rangeStringMonitor)):
            try:
                gpu_pid = row[1] + pid_offset
                outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"name\":\"%s\",\"ts\":\"%s\",\"dur\":\"%s\",\"ph\":\"X\",\"args\":{\"kernels\":\"%s\"}}\n"%(gpu_pid, row[2], f'Graph {row[0]}', row[3], row[4], row[5]))
            except ValueError:
                pass
    except:
        pass

    # Output APIs
    for row in connection.execute("select A.string as apiName, B.string as args, pid, tid, rocpd_api.start/1000.0, (rocpd_api.end-rocpd_api.start) / 1000.0, (rocpd_api.end != rocpd_api.start) as has_duration from rocpd_api INNER JOIN rocpd_string A on A.id = rocpd_api.apiName_id INNER Join rocpd_string B on B.id = rocpd_api.args_id %s order by rocpd_api.id"%(rangeStringApi)):
        try:
            adj_pid = row[2] + pid_offset
            if row[0]=="UserMarker":
                if row[6] == 0:  # instantaneous "mark" messages
                    outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"name\":\"%s\",\"ts\":\"%s\",\"ph\":\"i\",\"s\":\"p\",\"args\":{\"desc\":\"%s\"}}\n"%(adj_pid, row[3], row[1].replace('"',''), row[4], row[1].replace('"','')))
                else:
                    outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"name\":\"%s\",\"ts\":\"%s\",\"dur\":\"%s\",\"ph\":\"X\",\"args\":{\"desc\":\"%s\"}}\n"%(adj_pid, row[3], row[1].replace('"',''), row[4], row[5], row[1].replace('"','')))
            else:
                outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"name\":\"%s\",\"ts\":\"%s\",\"dur\":\"%s\",\"ph\":\"X\",\"args\":{\"desc\":\"%s\"}}\n"%(adj_pid, row[3], row[0], row[4], row[5], row[1].replace('"','').replace('\t','')))
        except ValueError:
            pass

    # Output api->op linkage
    for row in connection.execute("select rocpd_api_ops.id, pid, tid, gpuId, queueId, rocpd_api.end/1000.0 - 2, rocpd_op.start/1000.0 from rocpd_api_ops INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id %s"%(rangeStringApi)):
        try:
            fromtime = row[5] if row[5] < row[6] else row[6]
            adj_pid = row[1] + pid_offset
            gpu_pid = row[3] + pid_offset
            # Use unique IDs per node to avoid conflicts
            link_id = row[0] + (node_id * 10000000)
            outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"cat\":\"api_op\",\"name\":\"api_op\",\"ts\":\"%s\",\"id\":\"%s\",\"ph\":\"s\"}\n"%(adj_pid, row[2], fromtime, link_id))
            outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"cat\":\"api_op\",\"name\":\"api_op\",\"ts\":\"%s\",\"id\":\"%s\",\"ph\":\"f\", \"bp\":\"e\"}\n"%(gpu_pid, row[4], row[6], link_id))
        except ValueError:
            pass

    # Output HSA APIs
    try:
        for row in connection.execute("select A.string as apiName, B.string as args, pid, tid, rocpd_hsaApi.start/1000.0, (rocpd_hsaApi.end-rocpd_hsaApi.start) / 1000.0 from rocpd_hsaApi INNER JOIN rocpd_string A on A.id = rocpd_hsaApi.apiName_id INNER Join rocpd_string B on B.id = rocpd_hsaApi.args_id %s order by rocpd_hsaApi.id"%(rangeStringApi)):
            try:
                adj_pid = row[2] + pid_offset
                outfile.write(",{\"pid\":\"%s\",\"tid\":\"%s\",\"name\":\"%s\",\"ts\":\"%s\",\"dur\":\"%s\",\"ph\":\"X\",\"args\":{\"desc\":\"%s\"}}\n"%(adj_pid, row[3]+1, row[0], row[4], row[5], row[1].replace('"','')))
            except ValueError:
                pass
    except:
        pass

    #
    # Counters
    #

    # Counters should extend to the last event in the trace
        #
    # Counters
    #

    # Counters should extend to the last event in the trace
    T_end = 0
    for row in connection.execute("SELECT max(end)/1000 from (SELECT end from rocpd_api UNION ALL SELECT end from rocpd_op)"):
        T_end = int(row[0])
    if args.end:
        T_end = end_time

    # Loop over GPU for per-gpu counters
    gpuIdsPresent = []
    for row in connection.execute("SELECT DISTINCT gpuId FROM rocpd_op"):
        gpuIdsPresent.append(row[0])

    for gpuId in gpuIdsPresent:
        gpu_pid = gpuId + pid_offset
        
        # Create the queue depth counter
        depth = 0
        idle = 1
        for row in connection.execute("select * from (select rocpd_api.start/1000.0 as ts, \"1\" from rocpd_api_ops INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id AND rocpd_op.gpuId = %s %s UNION ALL select rocpd_op.end/1000.0, \"-1\" from rocpd_api_ops INNER JOIN rocpd_api on rocpd_api_ops.api_id = rocpd_api.id INNER JOIN rocpd_op on rocpd_api_ops.op_id = rocpd_op.id AND rocpd_op.gpuId = %s %s) order by ts"%(gpuId, rangeStringOp, gpuId, rangeStringOp)):
            try:
                if idle and int(row[1]) > 0:
                    idle = 0
                    outfile.write(',{"pid":"%s","name":"Idle","ph":"C","ts":%s,"args":{"idle":%s}}\n'%(gpu_pid, row[0], idle))
                if depth == 1 and int(row[1]) < 0:
                    idle = 1
                    outfile.write(',{"pid":"%s","name":"Idle","ph":"C","ts":%s,"args":{"idle":%s}}\n'%(gpu_pid, row[0], idle))
                depth = depth + int(row[1])
                outfile.write(',{"pid":"%s","name":"QueueDepth","ph":"C","ts":%s,"args":{"depth":%s}}\n'%(gpu_pid, row[0], depth))
            except ValueError:
                pass
        if T_end > 0:
            outfile.write(',{"pid":"%s","name":"Idle","ph":"C","ts":%s,"args":{"idle":%s}}\n'%(gpu_pid, T_end, idle))
            outfile.write(',{"pid":"%s","name":"QueueDepth","ph":"C","ts":%s,"args":{"depth":%s}}\n'%(gpu_pid, T_end, depth))

    # Create SMI counters
    try:
        for row in connection.execute("select deviceId, monitorType, start/1000.0, value from rocpd_monitor %s"%(rangeStringMonitor)):
            device_pid = row[0] + pid_offset
            outfile.write(',{"pid":"%s","name":"%s","ph":"C","ts":%s,"args":{"%s":%s}}\n'%(device_pid, row[1], row[2], row[1], row[3]))
        # Output the endpoints of the last range
        for row in connection.execute("select distinct deviceId, monitorType, max(end)/1000.0, value from rocpd_monitor %s group by deviceId, monitorType"%(rangeStringMonitor)):
            device_pid = row[0] + pid_offset
            outfile.write(',{"pid":"%s","name":"%s","ph":"C","ts":%s,"args":{"%s":%s}}\n'%(device_pid, row[1], row[2], row[1], row[3]))
    except:
        print(f"Did not find SMI data for {node_prefix}")

    # Create "faux calling stack frame" on gpu ops traces
    stacks = {}          # Call stacks built from UserMarker entries. Key is 'pid,tid'
    currentFrame = {}    # "Current GPU frame" (id, name, start, end). Key is 'pid,tid'

    class GpuFrame:
        def __init__(self):
            self.id = 0
            self.name = ''
            self.start = 0
            self.end = 0
            self.gpus = []
            self.totalOps = 0

    for row in connection.execute("SELECT '0', start/1000.0, pid, tid, B.string as label, '','','', '' from rocpd_api INNER JOIN rocpd_string A on A.id = rocpd_api.apiName_id AND A.string = 'UserMarker' INNER JOIN rocpd_string B on B.id = rocpd_api.args_id AND rocpd_api.start/1000.0 != rocpd_api.end/1000.0 %s UNION ALL SELECT '1', end/1000.0, pid, tid, B.string as label, '','','', '' from rocpd_api INNER JOIN rocpd_string A on A.id = rocpd_api.apiName_id AND A.string = 'UserMarker' INNER JOIN rocpd_string B on B.id = rocpd_api.args_id AND rocpd_api.start/1000.0 != rocpd_api.end/1000.0 %s UNION ALL SELECT '2', rocpd_api.start/1000.0, pid, tid, '' as label, gpuId, queueId, rocpd_op.start/1000.0, rocpd_op.end/1000.0 from rocpd_api_ops INNER JOIN rocpd_api ON rocpd_api_ops.api_id = rocpd_api.id INNER JOIN rocpd_op ON rocpd_api_ops.op_id = rocpd_op.id %s ORDER BY start/1000.0 asc"%(rangeStringApi, rangeStringApi, rangeStringApi)):
        try:
            key = (row[2], row[3])    # Key is 'pid,tid'
            if row[0] == '0':  # Frame start
                if key not in stacks:
                    stacks[key] = []
                stack = stacks[key].append((row[1], row[4]))

            elif row[0] == '1':  # Frame end
                if key in stacks and len(stacks[key]) > 0:
                    completed = stacks[key].pop()

            elif row[0] == '2':  # API + Op
                if key in stacks and len(stacks[key]) > 0:
                    frame = stacks[key][-1]
                    gpuFrame = None
                    if key not in currentFrame:    # First op under the current api frame
                        gpuFrame = GpuFrame()
                        gpuFrame.id = frame[0]
                        gpuFrame.name = frame[1]
                        gpuFrame.start = row[7]
                        gpuFrame.end = row[8]
                        gpuFrame.gpus.append((row[5] + pid_offset, row[6]))
                        gpuFrame.totalOps = 1
                    else:
                        gpuFrame = currentFrame[key]
                        # Another op under the same frame -> union them (but only if they are close together)
                        if gpuFrame.id == frame[0] and gpuFrame.name == frame[1] and (abs(row[7] - gpuFrame.end) < 200 or abs(gpuFrame.start - row[8]) < 200): 
                            if row[7] < gpuFrame.start: gpuFrame.start = row[7]
                            if row[8] > gpuFrame.end: gpuFrame.end = row[8] 
                            if (row[5] + pid_offset, row[6]) not in gpuFrame.gpus: 
                                gpuFrame.gpus.append((row[5] + pid_offset, row[6]))
                            gpuFrame.totalOps = gpuFrame.totalOps + 1

                        else:    # This is a new frame - dump the last and make new
                            gpuFrame = currentFrame[key]
                            for dest in gpuFrame.gpus:
                                outfile.write(',{"pid":"%s","tid":"%s","name":"%s","ts":"%s","dur":"%s","ph":"X","args":{"desc":"%s"}}\n'%(dest[0], dest[1], gpuFrame.name.replace('"',''), gpuFrame.start - 1, gpuFrame.end - gpuFrame.start + 1, f"UserMarker frame: {gpuFrame.totalOps} ops"))
                            currentFrame.pop(key)

                            # make the first op under the new frame
                            gpuFrame = GpuFrame()
                            gpuFrame.id = frame[0]
                            gpuFrame.name = frame[1]
                            gpuFrame.start = row[7]
                            gpuFrame.end = row[8]
                            gpuFrame.gpus.append((row[5] + pid_offset, row[6]))
                            gpuFrame.totalOps = 1

                    currentFrame[key] = gpuFrame

        except ValueError:
            pass

    connection.close()
    print(f"Finished processing {node_prefix}")


# Main execution: Process all input files
for idx, rpd_file in enumerate(args.input_rpd):
    node_id = idx  # Node 0, Node 1, etc.
    try:
        process_rpd_file(rpd_file, node_id, outfile, args)
    except Exception as e:
        print(f"Error processing {rpd_file}: {e}")
        import traceback
        traceback.print_exc()

# Close the JSON output
outfile.write("]\n")

if args.format == "object":
    outfile.write("} \n")

outfile.close()

print(f"\n{'='*80}")
print(f"Merged trace written to: {args.output_json}")
print(f"Total files processed: {len(args.input_rpd)}")
print(f"{'='*80}")
