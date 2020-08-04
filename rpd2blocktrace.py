# Format sqlite trace data as sjon for chrome:tracing

import sys
import os
import csv
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
import argparse
from enum import Enum

parser = argparse.ArgumentParser(description='convert RPD to json for chrome tracing')
parser.add_argument('input_rpd', type=str, help="input rpd db")
parser.add_argument('output_json', type=str, help="chrome tracing json output")
parser.add_argument('--start', type=int, help="start timestamp")
parser.add_argument('--end', type=int, help="end timestamp")
args = parser.parse_args()

print(args)

class block_apis(Enum):
    HipMalloc = 0
    BlockAlloc = 1
    BlockDealloc = 2
    InsertEvents = 3
    ProcessEvents = 4
    ProcessEventsSynchronizeEvent = 5
    ProcessEventsRetireEvent = 6
    BlockDeactivate = 7
    HipFree = 8

#block_apis = (
#    'HipMalloc',
#    'BlockAlloc',
#    'BlockDealloc',
#    'InsertEvents',
#    'ProcessEvents',
#    'ProcessEventsSynchronizeEvent',
#    'ProcessEventsRetireEvent',
#    'BlockDeactivate',
#    'HipFree',
#)

connection = sqlite3.connect(args.input_rpd)
outfile = open(args.output_json, 'w', encoding="utf-8")

outfile.write("[ {}\n");

def combine_pid_tid(pid, tid):
  return str(pid) + " Thread " + str(tid)

# label some rows
# tid becomes pid
# api index becomes tid
for row in connection.execute("select distinct pid, tid from block_api"):
    for api in block_apis:
        try:
            offset = api.value
            outfile.write(',{"name":"thread_name","ph":"M","pid":"%s","tid":"%s","args":{"name":"%s"}}\n'%(combine_pid_tid(row[0], row[1]), offset, api.name))
            outfile.write(',{"name":"thread_sort_index","ph":"M","pid":"%s","tid":"%s","args":{"sort_index":"%s"}}\n'%(combine_pid_tid(row[0], row[1]), offset, offset * 2))
        except ValueError:
            outfile.write("")

# Output block apis
for row in connection.execute("select blockApi as apiName, B.string as args, pid, tid, A.start/1000, (A.end-A.start) / 1000 from block_api A INNER JOIN rocpd_string B on B.id = A.args_id order by A.id"):
    try:
        duration = row[5] if row[5] > 0 else 1
        outfile.write(',{"pid":"%s","tid":"%s","name":"%s","ts":"%s","dur":"%s","ph":"X","args":{"desc":"%s"}}\n'%(combine_pid_tid(row[2], row[3]), block_apis[row[0]].value, row[0], row[4], duration, row[1].replace('"','')))
    except ValueError:
        outfile.write("")
        print("bad block_api row")


# arrows == cowbell


# Add arrows for block-lifecycle events
prevBlock = None
prevTime = None
prevPid = None
prevTid = None

for row in connection.execute("select A.id, pid, tid, (start + end) / 2000, blockApi, block from block_api A join rocpd_string B on B.id = A.args_id where B.string not like '%event=%' order by block, start asc"):
    pid = combine_pid_tid(row[1], row[2])
    tid = block_apis[row[4]].value
    time = row[3]
    if ((row[5] == prevBlock) and # only connect same block's lifecycle events
        tid != block_apis.HipMalloc.value and # don't draw arrows to reused blocks 
        (tid != block_apis.BlockAlloc.value or prevTid == block_apis.HipMalloc.value) and # don't draw arrows to BlockAlloc for reused blocks, unless it's from a fresh HipMalloc
        tid != block_apis.BlockDealloc.value): # don't draw arrows from BlockAlloc to BlockDealloc due to long time separation (available in "block_extra")
        outfile.write(',{"pid":"%s","tid":"%s","cat":"block","name":"Block lifecycle","ts":"%s","id":"%s","ph":"s"}\n'%(prevPid, prevTid, prevTime, row[0]))
        outfile.write(',{"pid":"%s","tid":"%s","cat":"block","name":"Block lifecycle","ts":"%s","id":"%s","ph":"f", "bp":"e"}\n'%(pid, tid, time, row[0]))
    prevBlock = row[5]
    prevTime = time
    prevPid = pid
    prevTid = tid


# Add arrows for extra block-lifecycle events
prevBlock = None
prevTime = None
prevPid = None
prevTid = None

for row in connection.execute("select A.id, pid, tid, (start + end) / 2000, blockApi, block from block_api A join rocpd_string B on B.id = A.args_id where B.string not like '%event=%' order by block, start asc"):
    pid = combine_pid_tid(row[1], row[2])
    tid = block_apis[row[4]].value
    time = row[3]
    if ((row[5] == prevBlock) and ( # only connect same block's lifecycle events
       tid == block_apis.HipMalloc.value or # draw arrows to reused blocks 
       (tid == block_apis.BlockAlloc.value and prevTid != block_apis.HipMalloc.value) or # draw arrows to BlockAlloc for reused blocks, unless it's from a fresh HipMalloc
       tid == block_apis.BlockDealloc.value)): # draw arrows from BlockAlloc to BlockDealloc
        outfile.write(',{"pid":"%s","tid":"%s","cat":"block_extra","name":"Block lifecycle","ts":"%s","id":"%s","ph":"s"}\n'%(prevPid, prevTid, prevTime, row[0]))
        outfile.write(',{"pid":"%s","tid":"%s","cat":"block_extra","name":"Block lifecycle","ts":"%s","id":"%s","ph":"f", "bp":"e"}\n'%(pid, tid, time, row[0]))
    prevBlock = row[5]
    prevTime = time
    prevPid = pid
    prevTid = tid


# Add arrows for event-lifecycle events
prevEvent = None
prevTime = None
prevPid = None
prevTid = None

for row in connection.execute("select A.id, pid, tid, (start + end) / 2000, blockApi, event from block_api A join rocpd_string B on B.id = A.args_id where B.string like '%event=%' order by event, start asc"):
    pid = combine_pid_tid(row[1], row[2])
    tid = block_apis[row[4]].value
    time = row[3]
    if ((row[5] == prevEvent) and # only connect same event's lifecycle events
        tid != block_apis.InsertEvents.value): # don't draw arrows to reused events
        if (tid == block_apis.ProcessEvents.value and prevTid == block_apis.ProcessEvents.value): # don't draw arrows from one ProcessEvents to the next; instead draw from first ProcessEvents to next lifecycle stage
            continue
        outfile.write(',{"pid":"%s","tid":"%s","cat":"event","name":"Event lifecycle","ts":"%s","id":"%s","ph":"s"}\n'%(prevPid, prevTid, prevTime, row[0]))
        outfile.write(',{"pid":"%s","tid":"%s","cat":"event","name":"Event lifecycle","ts":"%s","id":"%s","ph":"f", "bp":"e"}\n'%(pid, tid, time, row[0]))
    prevEvent= row[5]
    prevTime = time
    prevPid = pid
    prevTid = tid


# Add arrows for extra event-lifecycle events
prevEvent = None
prevTime = None
prevPid = None
prevTid = None

for row in connection.execute("select A.id, pid, tid, (start + end) / 2000, blockApi, event from block_api A join rocpd_string B on B.id = A.args_id where B.string like '%event=%' order by event, start asc"):
    pid = combine_pid_tid(row[1], row[2])
    tid = block_apis[row[4]].value
    time = row[3]
    if ((row[5] == prevEvent) and # only connect same event's lifecycle events
        (tid == block_apis.ProcessEvents.value and prevTid == block_apis.ProcessEvents.value)): # draw arrows from one ProcessEvents to the next
        outfile.write(',{"pid":"%s","tid":"%s","cat":"event_extra","name":"Event lifecycle","ts":"%s","id":"%s","ph":"s"}\n'%(prevPid, prevTid, prevTime, row[0]))
        outfile.write(',{"pid":"%s","tid":"%s","cat":"event_extra","name":"Event lifecycle","ts":"%s","id":"%s","ph":"f", "bp":"e"}\n'%(pid, tid, time, row[0]))
    prevEvent= row[5]
    prevTime = time
    prevPid = pid
    prevTid = tid


#### COUNTERS ####

# Counters should extend to the last event in the trace.  This means they need to have a value at Tend.
# Figure out when that is

T_end = 0
for row in connection.execute("SELECT max(end)/1000 from (SELECT end from rocpd_api UNION ALL SELECT end from rocpd_op UNION ALL SELECT end from block_api)"):
    T_end = int(row[0])


#Create HIP allocation counters
currentHipAllocations = 0
totalHipAllocatedSize = 0
for row in connection.execute("SELECT start/1000 as ts, blockApi, block, size, '1' FROM block_api WHERE blockApi='HipMalloc' UNION ALL SELECT end/1000, blockApi, block, size, '-1' FROM block_api WHERE blockApi='HipFree' ORDER BY ts asc"):
    try:
        currentHipAllocations = currentHipAllocations + int(row[4])
        if row[4] == '1': # hipMalloc
            totalHipAllocatedSize = totalHipAllocatedSize + row[3]
        else:             # hipFree
            totalHipAllocatedSize = totalHipAllocatedSize - row[3]
        assert(totalHipAllocatedSize >= 0)
        outfile.write(',{"pid":"HIP Counters","name":"HIP Allocated Memory","ph":"C","ts":%s,"args":{"size":%s}}\n'%(row[0],totalHipAllocatedSize))
        outfile.write(',{"pid":"HIP Counters","name":"Current HIP Allocations","ph":"C","ts":%s,"args":{"count":%s}}\n'%(row[0],currentHipAllocations))
    except ValueError:
        outfile.write("")


#Create caching allocator allocation counters
currentAllocatedBlocks = 0
totalAllocatedBlockSize = 0
currentActiveBlocks = 0
totalActiveBlockSize = 0
for row in connection.execute("SELECT start/1000 as ts, blockApi, block, size, '1' FROM block_api WHERE blockApi='BlockAlloc' UNION ALL SELECT start/1000, blockApi, block, size, '-1' FROM block_api WHERE blockApi='BlockDealloc' UNION ALL SELECT start/1000, blockApi, block, size, '-1' FROM block_api WHERE blockApi='BlockDeactivate' ORDER BY ts asc "):
    try:
        if row[4] == '1': # BlockAlloc 
            currentAllocatedBlocks = currentAllocatedBlocks + 1 
            totalAllocatedBlockSize = totalAllocatedBlockSize + row[3]
            currentActiveBlocks = currentActiveBlocks + 1
            totalActiveBlockSize = totalActiveBlockSize + row[3]
        elif row[4] == '-1':
            if row[1] == 'BlockDealloc':
                currentAllocatedBlocks = currentAllocatedBlocks - 1 
                totalAllocatedBlockSize = totalAllocatedBlockSize - row[3]
                assert(totalAllocatedBlockSize >= 0)
            elif row[1] == 'BlockDeactivate':
                currentActiveBlocks = currentActiveBlocks - 1
                totalActiveBlockSize = totalActiveBlockSize - row[3]
                assert(totalActiveBlockSize >= 0)
        assert(totalActiveBlockSize - totalAllocatedBlockSize >= 0)
        outfile.write(',{"pid":"Caching Allocator Counters","name":"Allocated Memory","ph":"C","ts":%s,"args":{"size":%s}}\n'%(row[0],totalAllocatedBlockSize))
        outfile.write(',{"pid":"Caching Allocator Counters","name":"Allocated Blocks","ph":"C","ts":%s,"args":{"count":%s}}\n'%(row[0],currentAllocatedBlocks))
        outfile.write(',{"pid":"Caching Allocator Counters","name":"Active Memory","ph":"C","ts":%s,"args":{"size":%s}}\n'%(row[0],totalActiveBlockSize))
        outfile.write(',{"pid":"Caching Allocator Counters","name":"Active Blocks","ph":"C","ts":%s,"args":{"count":%s}}\n'%(row[0],currentActiveBlocks))
    except ValueError:
        outfile.write("")


#Create caching allocator event counters
currentOutstandingEvents = 0
lastProcessedEvent = ""
lastProcessedEventTimestamp = 0
timeSinceLastProcessedEvent = 0
for row in connection.execute("SELECT A.start/1000 as ts, A.blockApi, A.block, A.size, A.event, '1' FROM block_api A JOIN rocpd_string B on B.id = A.args_id WHERE (A.blockApi = 'InsertEvents' AND B.string like '%stream=%') UNION ALL SELECT start/1000, blockApi, block, size, event, '0' FROM block_api WHERE (blockApi='ProcessEvents' OR blockApi='ProcessEventsSynchronizeEvent') UNION ALL SELECT start/1000, blockApi, block, size, event, '-1' FROM block_api WHERE blockApi='ProcessEventsRetireEvent' ORDER BY ts asc "):
    try:
        if row[5] == '1':   # InsertEvents
            currentOutstandingEvents = currentOutstandingEvents + 1
        elif row[5] == '0': # ProcessEvents or ProcessEventsSynchronizeEvent
            if (row[4] != lastProcessedEvent): # new event being processed
                lastProcessedEvent = row[4]
                timeSinceLastProcessedEvent = 0 # reset time counter
                lastProcessedEventTimestamp = row[0] # reset timestamp
        else:               # ProcessEventsRetireEvent
            assert row[4] == lastProcessedEvent, "lastProcessedEvent: " + lastProcessedEvent + ", eventBeingRetired: " + row[4]
            currentOutstandingEvents = currentOutstandingEvents - 1
            assert currentOutstandingEvents >= 0, "Current event being retired: " + row[4]
            timeSinceLastProcessedEvent = row[0] - lastProcessedEventTimestamp
            outfile.write(',{"pid":"Caching Allocator Counters","name":"Processing Time","ph":"C","ts":%s,"args":{"duration":%s}}\n'%(lastProcessedEventTimestamp,timeSinceLastProcessedEvent))
            outfile.write(',{"pid":"Caching Allocator Counters","name":"Processing Time","ph":"C","ts":%s,"args":{"duration":%s}}\n'%(row[0],timeSinceLastProcessedEvent))
        outfile.write(',{"pid":"Caching Allocator Counters","name":"Outstanding Events","ph":"C","ts":%s,"args":{"count":%s}}\n'%(row[0],currentOutstandingEvents))
    except ValueError:
        outfile.write("")

# Write out counter values till end of trace
if T_end > 0:
    outfile.write(',{"pid":"HIP Counters","name":"HIP Allocated Memory","ph":"C","ts":%s,"args":{"size":%s}}\n'%(T_end,totalHipAllocatedSize))
    outfile.write(',{"pid":"HIP Counters","name":"Current HIP Allocations","ph":"C","ts":%s,"args":{"count":%s}}\n'%(T_end,currentHipAllocations))
    outfile.write(',{"pid":"Caching Allocator Counters","name":"Allocated Memory","ph":"C","ts":%s,"args":{"size":%s}}\n'%(T_end,totalAllocatedBlockSize))
    outfile.write(',{"pid":"Caching Allocator Counters","name":"Allocated Blocks","ph":"C","ts":%s,"args":{"count":%s}}\n'%(T_end,currentAllocatedBlocks))
    outfile.write(',{"pid":"Caching Allocator Counters","name":"Active Memory","ph":"C","ts":%s,"args":{"size":%s}}\n'%(T_end,totalActiveBlockSize))
    outfile.write(',{"pid":"Caching Allocator Counters","name":"Active Blocks","ph":"C","ts":%s,"args":{"count":%s}}\n'%(T_end,currentActiveBlocks))
    outfile.write(',{"pid":"Caching Allocator Counters","name":"Outstanding Events","ph":"C","ts":%s,"args":{"count":%s}}\n'%(T_end,currentOutstandingEvents))
    outfile.write(',{"pid":"Caching Allocator Counters","name":"Processing Time","ph":"C","ts":%s,"args":{"duration":%s}}\n'%(T_end,timeSinceLastProcessedEvent))

outfile.write("]\n")
outfile.close()
connection.close()


