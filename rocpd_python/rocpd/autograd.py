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

# Create subclass table for autograd operators
#

import sys
import os
import re
import argparse
import sqlite3
from rocpd.importer import RocpdImportData
from rocpd.strings import cleanStrings
from rocpd.call_stacks import createCallStackTable, generateCallStacks
from rocpd.metadata import Metadata


def generateAutograd(imp):
    meta = Metadata(imp)
    if meta.get("Autograd::Generated") != None:
        raise Exception("Autograd data has already been generated")

    # dedupe string table before we start
    cleanStrings(imp, False) # yes False

    agStringId = importData.getStringId("PytorchAutograd")
    #importData.commitStrings()

    #Upgrade "UserMarker" calls to "PytorchAutograd" calls
    imp.connection.execute('UPDATE rocpd_api set apiName_id = ? where id in (select A.id  from rocpd_api A join rocpd_string B on B.id = A.args_id where B.string like "%, op_id = %" OR B.string like "%, seq = %" OR B.string like "%, sizes = %" OR B.string like "%, input_op_ids = %")', (agStringId, ))

    count = 0
    ag_inserts = [] # rows to bulk insert

    def commitRecords():
        nonlocal ag_inserts
        imp.commitStrings()
        imp.connection.executemany("INSERT INTO ext_autogradapi (api_ptr_id, autogradName_id, seq, op_id, sizes, input_op_ids) values (?, ?, ? ,? ,? ,?)", ag_inserts)
        imp.connection.commit()
        ag_inserts = []

    for row in imp.connection.execute("SELECT A.id, B.string FROM rocpd_api A JOIN rocpd_string B ON B.id = A.args_id WHERE A.apiName_id = ?", (agStringId, )):
        if (count % 100000 == 99999):
            commitRecords()

        # all params
        m = re.match('^(.+), seq = ([0-9]+), op_id = ([0-9]+), sizes = (.*), input_op_ids = (.*)$', row[1])
        if m:
            ag_inserts.append((row[0], importData.getStringId(m.group(1)), m.group(2), m.group(3), m.group(4),  m.group(5)))
            continue
        # all but seq
        m = re.match('^(.+), op_id = ([0-9]+), sizes = (.*), input_op_ids = (.*)$', row[1])
        if m:
            ag_inserts.append((row[0], importData.getStringId(m.group(1)), "", m.group(2), m.group(3), m.group(4)))
            continue
        # seq and ops (size reporting not enabled)
        m = re.match('^(.+), seq = ([0-9]+), op_id = ([0-9]+)$', row[1])
        if m:
            ag_inserts.append((row[0], importData.getStringId(m.group(1)), m.group(2), m.group(3), "",""))
            continue
        # just op_id
        m = re.match('^(.+), op_id = ([0-9]+)$', row[1])
        if m:
            ag_inserts.append((row[0], importData.getStringId(m.group(1)), "", m.group(2), "", ""))
            continue
        # Patterns we missed
        print(f"OOPS: {row[0]} : {row[1]}")

    commitRecords()

    # Replace rocpd_api.args with just the operator name instead of name(args)
    #imp.connection.execute('delete from rocpd_string where id in (select args_id from rocpd_api where apiName_id = ?)', (agStringId, )) # not safe, saves so much filesize, do it right?
    imp.connection.execute('update rocpd_api set args_id = (select autogradName_id from ext_autogradapi where ext_autogradapi.api_ptr_id = rocpd_api.id) where id in (select api_ptr_id from ext_autogradapi)')
    # Set calls back to UserMarker for now.  We can't reliably distinguish all of them from other markers
    umStringId = importData.getStringId("UserMarker")
    imp.connection.execute('UPDATE rocpd_api SET apiName_id = ? WHERE apiName_id = ?', (umStringId, agStringId))
    # mark metadata so we don't do this again
    meta.set("Autograd::Generated", "True")
    imp.connection.commit()

    # dedupe strings again... to remove unrefferenced strings from the above replacement
    cleanStrings(imp, False) 


def createAutogradTable(imp):
    meta = Metadata(imp)
    if meta.get("Autograd::Table") != None:
        raise Exception("Autograd table has already been created")

    # Create table
    imp.connection.execute('CREATE TABLE IF NOT EXISTS "ext_autogradapi" ("api_ptr_id" integer NOT NULL PRIMARY KEY REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "autogradName_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "seq" integer NOT NULL, "op_id" integer NOT NULL, "sizes" varchar(4096) NOT NULL, "input_op_ids" varchar(4096) NOT NULL)')

    # View to join in api strings
    imp.connection.execute('CREATE VIEW IF NOT EXISTS "autograd" AS SELECT B.id, pid, tid, start, end, C.string AS autogradName, seq, op_id, sizes, input_op_ids FROM ext_autogradapi A JOIN rocpd_api B ON B.id = A.api_ptr_id JOIN rocpd_string C ON C.id = A.autogradName_id')

    # Add metadata to indicate one of our columns referrence rocpd_string
    imp.connection.execute(""" INSERT INTO "rocpd_metadata" (tag, value) VALUES ('references::rocpd_string.id', '("ext_autogradapi", "autogradName_id")') """)
    meta.set("Autograd::Table", "True")


def createAutogradStackViews(imp):
    # Create views track kernels, cpu, gpu, size by autograd operator
    imp.connection.execute('CREATE VIEW IF NOT EXISTS _callstack_autograd_kernel as select A.*, B.sizes, B.autogradname_id, C.kernelname_id from ext_callstack A left join ext_autogradapi B on B.api_ptr_id = A.parent_id left join rocpd_kernelapi C on C.api_ptr_id = A.child_id')
    imp.connection.execute('CREATE VIEW IF NOT EXISTS _callstack_autograd_kernel_exclusive as select * from _callstack_autograd_kernel where depth=1 and kernelName_id not null')
    imp.connection.execute('CREATE VIEW IF NOT EXISTS _autogradKernel as select autogradName_id, kernelName_id, sizes, count(*) as calls, avg(gpu_time) as avg_gpu, sum(gpu_time) as total_gpu from _callstack_autograd_kernel_exclusive group by autogradName_id, kernelName_id, sizes')
    imp.connection.execute('CREATE VIEW IF NOT EXISTS autogradKernel as select B.string as autogradName, C.string as kernelName, sizes, calls, avg_gpu, total_gpu from _autogradKernel A join rocpd_string B on B.id = A.autogradName_id join rocpd_string C on C.id = A.kernelName_id')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Utility for creating a pytorch autograd api subclass')
    parser.add_argument('input_rpd', type=str, help="input rpd db")
    parser.add_argument('--skip_callstacks', action='store_true', help="Skip creation of the 'ext_callstack' table and views to track gpu activity by autograd operator")
    args = parser.parse_args()

    connection = sqlite3.connect(args.input_rpd)

    importData = RocpdImportData()
    importData.resumeExisting(connection) # load the current db state

    createAutogradTable(importData)
    generateAutograd(importData)

    if not args.skip_callstacks:
        createCallStackTable(importData)
        generateCallStacks(importData)
        createAutogradStackViews(importData)
