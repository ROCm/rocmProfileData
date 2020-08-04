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

from rocpd.importer import RocpdImportData

def deserializeApis(imp, srcApis):
    count = 0
    api_inserts = []    # rows to bulk insert
    op_inserts = []     # rows to bulk insert
    api_removes = []    # rows to bulk remove

    def commitRecords():
        nonlocal api_inserts
        nonlocal op_inserts
        nonlocal api_removes
        imp.commitStrings()
        imp.connection.executemany("insert into rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?,?)", api_inserts)
        imp.connection.executemany("insert into rocpd_op(id, gpuId, queueId, sequenceId, completionSignal, start, end, description_id, opType_id) values (?,?,?,?,?,?,?,?,?)", op_inserts)
        imp.connection.executemany("delete from rocpd_api where id = ?", api_removes)
        imp.connection.commit()
        api_inserts = []
        op_inserts = []
        api_removes = []

    for row in imp.connection.execute("select A.id, A.pid, A.tid, A.start, A.end, A.args_id, B.string, C.string from rocpd_api A join rocpd_string B on B.id = A.apiName_id join rocpd_string C on C.id = A.args_id where A.apiName_id in (select id from rocpd_string where string in (%s))" % str(srcApis)[1:-1]):
        args = {}
        for line in row[7].split(','):
            key, value = line.partition("=")[::2]
            args[key.strip()] = value.strip()

        if 's_api' in args:
            name_id = imp.getStringId(args['s_api'])
            api_inserts.append((imp.api_id, row[1], row[2], row[3], row[4], name_id, row[5]))
            api_removes.append((row[0],))
            imp.api_id = imp.api_id + 1
            count = count + 1

        if 's_op' in args:
            name_id = imp.getStringId(args['s_op'])
            gpuId = args['gpuId'] if 'gpuId' in args else '0'
            queueId = args['queueId'] if 'queueId' in args else '0' 
            sequenceId = args['sequenceId'] if 'sequenceId' in args else '0'
            completionSignal = args['completionSignal'] if 'completionSignal' in args else ''
            start = args['start'] if 'start' in args else row[3]
            end = args['end'] if 'end' in args else row[4]
            op_inserts.append((imp.op_id, gpuId, queueId, sequenceId, completionSignal, start, end, row[5], name_id))
            api_removes.append((row[0],))
            imp.op_id = imp.op_id + 1
            count = count + 1

        if (count % 100000 == 99999):
            commitRecords()
    commitRecords()


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Promote roctx serialized ops to actual ops')
  parser.add_argument('input_rpd', type=str, help="input rpd db")
  args = parser.parse_args()

  connection = sqlite3.connect(args.input_rpd)

  importData = RocpdImportData()
  importData.resumeExisting(connection)	# load the current db state

  roctxApis = ["UserMarker"]
  print(f"Deserializing apis in: {str(roctxApis)[1:-1]}")
  deserializeApis(importData, roctxApis)
