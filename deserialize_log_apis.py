# Elevate roctx logging messages to 1st class api events
#
# Logged messages should be formatted as: api=<> args1=<> arg2=<>
#

import sys
import os
import csv
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
import argparse

def deserializeApis(connection, srcApis):
    print(f"Deserializing apis in: {str(srcApis)[1:-1]}")

    count = 0
    api_inserts = []    # rows to bulk insert
    string_inserts = [] # rows to bulk insert

    global strings
    global string_id
    global api_id

    def commitRecords():
        nonlocal api_inserts
        nonlocal string_inserts
        connection.executemany("insert into rocpd_string(id, string) values (?,?)", string_inserts)
        connection.executemany("insert into rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?,?)", api_inserts)
        connection.commit()
        api_inserts = []
        string_inserts = []

    for row in connection.execute("select A.id, A.pid, A.tid, A.start, A.end, A.args_id, B.string, C.string from rocpd_api A join rocpd_string B on B.id = A.apiName_id join rocpd_string C on C.id = A.args_id where A.apiName_id in (select id from rocpd_string where string in (%s))" % str(srcApis)[1:-1]):
        args = {}
        for line in row[7].split(','):
            key, value = line.partition("=")[::2]
            args[key.strip()] = value.strip()
        if 'op' in args:
            opname = args['op']
            try:
                name_id = strings[opname]
            except:
                strings[opname] = string_id
                string_inserts.append((string_id, opname))
                name_id = string_id
                string_id = string_id + 1
            api_inserts.append((api_id, row[1], row[2], row[3], row[4], name_id, row[5]))
            api_id = api_id + 1
            count = count + 1
        if (count % 100000 == 99999):
            commitRecords()
    commitRecords()


def deleteApis(connection, srcApis):
    connection.execute("delete from rocpd_api where apiName_id in (select id from rocpd_string where string in (%s))" % str(srcApis)[1:-1])
    connection.commit()



def buildStringCache(connection):
    global empty_string_id
    global string_id
    global strings

    #FIXME: ensure it is present
    for row in connection.execute("select id from rocpd_string where string=''"):
        empty_string_id = row[0]
    
    for row in connection.execute("select id from rocpd_string order by id desc limit 1"):
        string_id = row[0] + 1

    for row in connection.execute("select id, string from rocpd_string"):
        strings[row[0]] = row[1]

def buildCurrentIds(connection):
    global op_id
    global api_id
    for row in connection.execute("select id from rocpd_op order by id desc limit 1"):
        op_id = row[0] + 1
    for row in connection.execute("select id from rocpd_api order by id desc limit 1"):
        api_id = row[0] + 1


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Promote roctx serialized ops to actual ops')
  parser.add_argument('input_rpd', type=str, help="input rpd db")
  parser.add_argument('--start', type=int, help="start timestamp")
  parser.add_argument('--end', type=int, help="end timestamp")
  args = parser.parse_args()

  connection = sqlite3.connect(args.input_rpd)

  # Initialize data to current db state
  # FIXME: Move all this someplace reusable

  #Set up primary keys
  string_id = 1
  op_id = 1
  api_id = 1
  hsa_id = 1

  # Dicts
  strings = {}    # string -> id

  # Empty string
  empty_string_id = 1

  buildStringCache(connection)
  buildCurrentIds(connection)

  roctxApis = ["UserMarker"]
  deserializeApis(connection, roctxApis)
  deleteApis(connection, roctxApis)


