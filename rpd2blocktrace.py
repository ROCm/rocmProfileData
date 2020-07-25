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

block_apis = (
    'BlockAlloc',
    'BlockFreeDeallocate',
    'BlockInsertEvents',
    'ProcessEvents',
    'BlockFreeDeactivate',
)

connection = sqlite3.connect(args.input_rpd)
outfile = open(args.output_json, 'w', encoding="utf-8")

outfile.write("[ {}\n");


# label some rows
# tid becomes pid
# api index becomes tid
for row in connection.execute("select distinct tid from block_api"):
    for api in block_apis:
        try:
            offset = block_apis.index(api)
            outfile.write(',{"name":"thread_name","ph":"M","pid":"%s","tid":"%s","args":{"name":"%s"}}\n'%(row[0], offset, api))
            outfile.write(',{"name":"thread_sort_index","ph":"M","pid":"%s","tid":"%s","args":{"sort_index":"%s"}}\n'%(row[0], offset, offset * 2))
        except ValueError:
            outfile.write("")

# Output block apis
for row in connection.execute("select blockApi as apiName, B.string as args, pid, tid, A.start/1000, (A.end-A.start) / 1000 from block_api A INNER JOIN rocpd_string B on B.id = A.args_id order by A.id"):
    try:
        duration = row[5] if row[5] > 0 else 1
        outfile.write(',{"pid":"%s","tid":"%s","name":"%s","ts":"%s","dur":"%s","ph":"X","args":{"desc":"%s"}}\n'%(row[3], block_apis.index(row[0]), row[0], row[4], duration, row[1].replace('"','')))
    except ValueError:
        outfile.write("")
        print("bad block_api row")


# arrows == cowbell


prevBlock = None
prevTime = None
prevPid = None
prevTid = None

for row in connection.execute("select A.id, tid, (start + end) / 2000, blockApi, block from block_api A join rocpd_string B on B.id = A.args_id where B.string not like '%stream%' order by block, start asc"):
    pid = row[1]
    tid = block_apis.index(row[3])
    time = row[2]
    if (row[4] == prevBlock):
        outfile.write(',{"pid":"%s","tid":"%s","cat":"api_op","name":"api_op","ts":"%s","id":"%s","ph":"s"}\n'%(prevPid, prevTid, prevTime, row[0]))
        outfile.write(',{"pid":"%s","tid":"%s","cat":"api_op","name":"api_op","ts":"%s","id":"%s","ph":"f", "bp":"e"}\n'%(pid, tid, time, row[0]))
    prevBlock = row[4]
    prevTime = time
    prevPid = pid
    prevTid = tid


outfile.write("]\n")
outfile.close()
connection.close()

