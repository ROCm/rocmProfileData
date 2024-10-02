################################################################################
# Copyright (c) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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
# Trim any data from an rpd file that doesn't fall within a specified time span.
#

import sqlite3
import argparse

parser = argparse.ArgumentParser(description='Permanently remove events/data from an RPD file that falls outside a specified time range.  Range start and end values default to the files\' original start and end')
parser.add_argument('input_rpd', type=str, help="input rpd db")
parser.add_argument('--start', type=str, help="start time - default ns or percentage %%. Number only is interpreted as ns. Number with %% is interpreted as percentage. Number with leading '+' is interpreted as delta from the start time.")
parser.add_argument('--end', type=str, help="end time - default ns or percentage %%. See help for --start")
parser.add_argument('--dryrun', action=argparse.BooleanOptionalAction, help="compute range but take no action")
args = parser.parse_args()

connection = sqlite3.connect(args.input_rpd)

min_time = connection.execute("select MIN(start) from rocpd_api;").fetchall()[0][0]
max_time = connection.execute("select MAX(end) from rocpd_api;").fetchall()[0][0]
if (min_time == None):
    raise Exception("Trace file is empty.")

print(f"\t duration: \t{(max_time-min_time) / 1000000000} seconds")

# Calculate trim start
if args.start:
    if "%" in args.start:
        start_time = int( (max_time - min_time) * ( int( args.start.replace("%","") )/100 ) + min_time )
    elif args.start.startswith('+'):
        start_time = int(args.start[1:]) + min_time
    else:
        start_time = int(args.start)
else:
    start_time = min_time

# Calculate trim end
if args.end:
    if "%" in args.end:
        end_time = int( (max_time - min_time) * ( int( args.end.replace("%","") )/100 ) + min_time )
    elif args.end.startswith('+'):
        end_time = int(args.end[1:]) + min_time
    else:
        end_time = int(args.end)
else:
    end_time = max_time

print("Timestamps:")
print(f"\t    first: \t{min_time} ns")
print(f"\t     last: \t{max_time} ns")
print(f"\trng_start: \t{start_time} ns")
print(f"\trng_end  : \t{end_time} ns")

assert start_time >= min_time
assert start_time <= max_time
assert end_time >= min_time
assert end_time <= max_time

print()
print(f"Trimmed range:    {start_time} --> {end_time}")
print(f"Trimmed duration: {(end_time-start_time)/1000000000} seconds")

apiCount = connection.execute("select count(*) from rocpd_api").fetchall()[0][0]
apiRemoveCount = connection.execute("select count(*) from rocpd_api where start < %s or start > %s"%(start_time, end_time)).fetchall()[0][0]
opCount = connection.execute("select count(*) from rocpd_op").fetchall()[0][0]
opRemoveCount = connection.execute("select count(*) from rocpd_api A join rocpd_api_ops B on B.api_id = A.id where A.start < %s or A.start > %s"%(start_time, end_time)).fetchall()[0][0]
print()
print(f"Removing {apiRemoveCount} of {apiCount} api calls.  {apiCount - apiRemoveCount} remaining")
print(f"Removing {opRemoveCount} of {opCount} async ops.  {opCount - opRemoveCount} remaining")

if args.dryrun:
    print("Dry run, exiting")
    exit()

connection.execute("delete from rocpd_api where start < %s or start > %s"%(start_time, end_time))
connection.execute("delete from rocpd_api_ops where api_id not in (select id from rocpd_api)")
connection.execute("delete from rocpd_op where id not in (select op_id from rocpd_api_ops)")
try:
    connection.execute("delete from rocpd_monitor where start < (select min(start) from rocpd_api) or start > (select max(end) from rocpd_op)")
except:
    pass

connection.commit()

#clear any unused strings
stringCount = connection.execute("select count(*) from rocpd_string").fetchall()[0][0]
from rocpd.importer import RocpdImportData
from rocpd.strings import cleanStrings
importData = RocpdImportData()
importData.resumeExisting(connection) # load the current db state
cleanStrings(importData, False)
stringRemaingCount = connection.execute("select count(*) from rocpd_string").fetchall()[0][0]
print(f"Removed {stringCount - stringRemaingCount} of {stringCount} strings.  {stringRemaingCount} remaining")

connection.isolation_level = None
connection.execute("vacuum")
connection.commit()
connection.close()
