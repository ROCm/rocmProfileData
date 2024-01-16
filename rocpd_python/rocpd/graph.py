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

# Create graph tables
#

import sys
import os
import re
import argparse
import sqlite3
from rocpd.importer import RocpdImportData
from rocpd.strings import cleanStrings
from rocpd.metadata import Metadata
from rocpd.subclass import createSubclassTable


def generateGraph(imp):
    meta = Metadata(imp)
    if meta.get("Graph::Generated") != None:
        raise Exception("Graph data has already been generated")

    # create a subclass table for graph related calls - easier
    graphApis = [
        'hipStreamBeginCapture',
        'hipStreamEndCapture',
        'hipGraphInstantiate',
        'hipGraphInstantiateWithFlags',
        'cudaStreamBeginCapture_v10000',
        'cudaStreamBeginCapture_ptsz_v10000',
        'cudaStreamEndCapture_v10000',
        'cudaStreamEndCapture_ptsz_v10000',
        'cudaGraphInstantiate_v10000',
        'cudaGraphInstantiateWithFlags_v11040',
        #'hipGraphLaunch',
    ]

    argTypes = {
        'stream' : 'varchar(18) NOT NULL',
        'graph' : 'varchar(18) NOT NULL',
        'graphExec' : 'varchar(18) NOT NULL',
    }

    print(f"Creating subclass table for 'graph' apis: {graphApis}")
    createSubclassTable(importData, 'api', 'graph', graphApis, argTypes)

    # create a subclass table for graph launches
    graphApis = [
        'hipGraphLaunch',
        'cudaGraphLaunch_v10000',
        'cudaGraphLaunch_ptsz_v10000',
    ]

    print(f"Creating subclass table for 'graphLaunch' apis: {graphApis}")
    createSubclassTable(importData, 'api', 'graphLaunch', graphApis, argTypes)


    # Fill the graph list table
    # If something goes wrong look here.  Assume graph apis were called correctly by application

    imp.connection.execute('INSERT into "ext_graph"("graph","graphExec","stream","start","end") select B.graph, B.graphExec, stream, start, end from (select stream, graph, lag(start) over (order by stream,start asc) as start, end from (select * from rocpd_graphapi A join rocpd_api B on B.id = A.api_ptr_id where stream != "" order by stream,start asc)) A join (select graph, graphExec from rocpd_graphapi where graph != "" and graphExec != "") B on B.graph = A.graph');


    # Populate the graph -> kernel bridge table
    graphs = [];
    for row in imp.connection.execute('SELECT id, stream, start, end from ext_graph'):
        graphs.append(row)

    for row in graphs:
        imp.connection.execute('INSERT INTO ext_graph_kernelapis("graph_id","api_id", "sequence") SELECT %s, B.id, ROW_NUMBER() OVER (ORDER BY B.start) from rocpd_kernelApi A join rocpd_api B on B.id = A.api_ptr_id where A.stream = "%s" and B.end >= %s and B.end <= %s order by B.start'%(row[0],row[1],row[2],row[3],))


    # mark metadata so we don't do this again
    meta.set("Graph::Generated", "True")
    imp.connection.commit()



def createGraphTable(imp):
    meta = Metadata(imp)
    if meta.get("Graph::Table") != None:
        raise Exception("Graph table has already been created")

    # Create tables
    imp.connection.execute('CREATE TABLE IF NOT EXISTS "ext_graph" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "graph" varchar(18) NOT NULL, "graphExec" varchar(18) NOT NULL, "stream" varchar(18) NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL)')
    imp.connection.execute('CREATE TABLE IF NOT EXISTS "ext_graph_kernelapis" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "graph_id" integer NOT NULL REFERENCES "ext_graph" ("id") DEFERRABLE INITIALLY DEFERRED, "api_id" integer NOT NULL REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "sequence" integer NOT NULL)')

    # View to show graph launches (cpu and gpu time)
    imp.connection.execute('CREATE VIEW IF NOT EXISTS graphLaunch AS select B.id, pid, tid, E.string as apiName, A.graphExec, A.stream, D.gpuId, D.queueId, (B.end - B.start)/1000 as apiDuration_usec, (max(D.end) - min(D.start))/1000 as opDuration_usec, sum((D.end - D.start)/1000) as gpuTime_usec from rocpd_graphLaunchapi A join rocpd_api B on B.id = A.api_ptr_id join rocpd_api_ops C on C.api_id = A.api_ptr_id join rocpd_op D on D.id = C.op_id join rocpd_string E on B.apiName_id = E.id group by b.id')

     # View to show graph kernels
    imp.connection.execute('CREATE VIEW IF NOT EXISTS graphKernel as select graphExec, sequence, D.string as kernelName, gridX, gridY, gridZ, workgroupX, workgroupY, workgroupZ, groupSegmentSize, privateSegmentSize from ext_graph_kernelapis A join ext_graph B on B.id = A.graph_id join rocpd_kernelapi C on C.api_ptr_id = A.api_id join rocpd_string D on D.id = C.kernelName_id')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Utility for creating a graph api subclass')
    parser.add_argument('input_rpd', type=str, help="input rpd db")
    args = parser.parse_args()

    connection = sqlite3.connect(args.input_rpd)

    importData = RocpdImportData()
    importData.resumeExisting(connection) # load the current db state

    createGraphTable(importData)
    generateGraph(importData)
