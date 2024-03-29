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
# Create an rpd file from rocprofiler output files
#

import sys
import os
import csv
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
import argparse
from os import path
from pathlib import Path
from rocpd.schema import RocpdSchema
from rocpd.importer import RocpdImportData



def importOps(imp, infile):
    exp = re.compile("^(\d*):(\d*)\s+(\d*):(\d*)\s+(\w+):(\d*):(\d*).*$")
    count = 0;
    op_inserts = [] # rows to bulk insert
    api_ops_inserts = [] # rows to bulk insert

    def commitRecords():
        nonlocal op_inserts
        nonlocal api_ops_inserts
        imp.commitStrings()
        imp.connection.executemany("insert into rocpd_op(id, gpuId, queueId, sequenceId, completionSignal,  start, end, description_id, opType_id) values (?,?,?,'','',?,?,?,?)", op_inserts)
        imp.connection.executemany("insert into rocpd_api_ops(api_id, op_id) values (?,?)", api_ops_inserts)
        imp.connection.commit()
        op_inserts = []
        api_ops_inserts = []

    for line in infile:
        m = exp.match(line)
        if m:
            name = imp.getStringId(m.group(5))
            desc = imp.getStringId("")

            if len(m.group(6)) > 0:
                api_ops_inserts.append((int(m.group(6)), imp.op_id))

            op_inserts.append((imp.op_id, m.group(3), m.group(4), m.group(1), m.group(2), desc, name))
            imp.op_id = imp.op_id + 1
            count = count + 1
        if (count % 100000 == 99999):
            commitRecords()
    commitRecords()


def importApis(imp, infile):
    exp =    re.compile("^(\d*):(\d*)\s+(\d*):(\d*)\s+(\w+)\((.*)\)\s+:.*$")
    expfix = re.compile("^(\d*):(\d*)\s+(\d*):(\d*)\s+(\w+)\((.*)\)( kernel=.*)\s+:(\d*)$")
    count = 0
    api_inserts = [] # rows to bulk insert

    def commitRecords():
        nonlocal api_inserts
        imp.commitStrings()
        imp.connection.executemany("insert into rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?,?)", api_inserts)
        imp.connection.commit()
        api_inserts = []

    for line in infile:
        kernstring = None
        m = expfix.match(line)
        if m:
            kernstring = m.group(6) + ', ' + m.group(7)
        else:
            m = exp.match(line)
        if m:
            api = imp.getStringId(m.group(5))
            kernstring = kernstring or m.group(6)
            arg = imp.getStringId(kernstring)

            api_inserts.append((imp.api_id, m.group(3), m.group(4), m.group(1), m.group(2), api, arg))
            imp.api_id = imp.api_id + 1
            count = count + 1
        if (count % 100000 == 99999):
            commitRecords()
    commitRecords()


def importHsa(imp, infile):
    exp = re.compile("^(\d*):(\d*)\s+(\d*):(\d*)\s+(\w+)\((.*)\)\s*=\s*(\d*).*$")
    count = 0
    hsa_inserts = [] # rows to bulk insert

    def commitRecords():
        nonlocal hsa_inserts

def importRoctx(imp, infile):
    exp = re.compile("^(\d*)\s+(\d*):(\d*)\s+(\d+):\d+:\"(.*)\".*$")
    count = 0;
    stack = []
    api_inserts = [] # rows to bulk insert

    def commitRecords():
        nonlocal api_inserts
        imp.commitStrings()
        imp.connection.executemany("insert into rocpd_api(id, pid, tid, start, end, apiName_id, args_id) values (?,?,?,?,?,?,?)", api_inserts)
        imp.connection.commit()
        api_inserts = []

    for line in infile:
        m = exp.match(line)
        if m:
            api = imp.getStringId("UserMarker")
            arg = imp.getStringId(m.group(5))

            entryType = int(m.group(4))

            if entryType == 0:        # instantaneous marker
                api_inserts.append((imp.api_id, m.group(2), m.group(3), m.group(1), m.group(1), api, arg))
                imp.api_id = imp.api_id + 1
                count = count + 1
            if entryType == 1:
                stack.append((m.group(1), arg))

            if entryType == 2:
                entry = stack.pop()
                api_inserts.append((imp.api_id, m.group(2), m.group(3), entry[0], m.group(1), api, entry[1]))
                imp.api_id = imp.api_id + 1
                count = count + 1

        if (count % 100000 == 99999):
            commitRecords()
    commitRecords()


# Clear args for a given api
def clearApiArgs(imp, apiname):
    imp.connection.execute("update rocpd_api set args_id = ? where apiName_id in (select id from rocpd_string where string = ?)", (imp.empty_string_id, apiname))
    connection.commit()


# Remove unreferenced rows from string table
def purgeStrings(imp):
    imp.connection.execute("delete from rocpd_string where id not in (\
            select distinct apiName_id from rocpd_api \
            union all select distinct args_id from rocpd_api \
            union all select distinct description_id from rocpd_op \
            union all select distinct opType_id from rocpd_op \
            union all select distinct kernelName_id from rocpd_kernelapi \
            )")
            # DISABLED to review HSA support
            #union all select distinct apiName_id from rocpd_hsaApi
            #union all select distinct args_id from rocpd_hsaApi 
    connection.commit()


#
# Build op child classes for the following apis
#
# hipHccModuleLaunchKernel
# hipLaunchKernel
# hipExtModuleLaunchKernel

def populateKernelInfo(imp):
    kernelApis = ['hipHccModuleLaunchKernel', 'hipLaunchKernel', 'hipExtModuleLaunchKernel']
    print(f"Extracting kernel info for: {str(kernelApis)[1:-1]}")

    count = 0
    kernel_inserts = [] # rows to bulk insert

    def commitRecords():
        nonlocal kernel_inserts
        imp.commitStrings
        imp.connection.executemany("insert into rocpd_kernelapi(api_ptr_id, stream, gridX, gridY, gridZ, workgroupX, workgroupY, workgroupZ, groupSegmentSize, privateSegmentSize, codeObject_id, kernelName_id, kernelArgAddress, aquireFence, releaseFence) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", kernel_inserts)
        imp.connection.commit()
        kernel_inserts = []


    for row in imp.connection.execute("select A.api_id, C.string from rocpd_api_ops A join rocpd_api B on B.id = A.api_id join rocpd_string C on C.id = B.args_id where B.apiName_id in (select id from rocpd_string where string in (%s))" % str(kernelApis)[1:-1]):
        args = {}
        if ('kernel=' in row[1]):
            input_line, kernel_line = row[1].split('kernel=')
        else:
            input_line = row[1]
            kernel_line = None
        for line in input_line.split(','):
            key, value = line.partition("=")[::2]
            args[key.strip()] = value.strip()
        stream = args['stream'] if 'stream' in args else 0
        # Rocprof doesn't seem to populate "numBlocks={}, dimBlocks={}" at the moment
        gridx = args['gridDimX'] if 'gridDimX' in args else 0
        gridy = args['gridDimY'] if 'gridDimY' in args else 0
        gridz = args['gridDimZ'] if 'gridDimZ' in args else 0
        bdimx = args['blockDimX'] if 'blockDimX' in args else 0
        bdimy = args['blockDimY'] if 'blockDimY' in args else 0
        bdimz = args['blockDimZ'] if 'blockDimZ' in args else 0
        shmem = args['sharedMemBytes'] if 'sharedMemBytes' in args else 0
        prmem = 0
        #kernstring = args['kernel'] if 'kernel' in args else ''
        kernstring = kernel_line if (kernel_line != None) else ''
        kargs = args['args'] if 'args' in args else ''
        aqfence = '';
        relfence = '';
        kernel = imp.getStringId(kernstring)

        kernel_inserts.append((row[0], stream, gridx, gridy, gridz, bdimx, bdimy, bdimz, shmem, prmem, 0, kernel, kargs, aqfence, relfence))
        count = count + 1
        if (count % 100000 == 99999):
            commitRecords()
    commitRecords()
    # copy the kernel names into the base op table's description
    # Most use cases will just want the kernel name and can avoid joining KernelOp
    imp.connection.execute("CREATE INDEX opid_index ON rocpd_api_ops (op_id)");
    imp.connection.execute("update rocpd_op set description_id = (select kernelName_id from rocpd_api_ops A join rocpd_kernelapi B on B.api_ptr_id = A.api_id where rocpd_op.id = A.op_id) where rocpd_op.id in (select A.id from rocpd_api_ops A join rocpd_kernelapi B on B.api_ptr_id=A.api_id)")
    imp.connection.commit()


# Build op child classes for the following apis
# hipMemcpyWithStream
# hipMemsetAsync
# hipMemcpyAsync

def populateCopyInfo(imp):
    copyApis = ['hipMemcpy', 'hipMemcpy2D', 'hipMemcpy2DAsync', 'hipMemcpyAsync', 'hipMemcpyDtoD', 'hipMemcpyDtoDAsync', 'hipMemcpyDtoH', 'hipMemcpyDtoHAsync', 'hipMemcpyFromSymbol', '.hipMemcpyFromSymbolAsync', 'hipMemcpyHtoD', 'hipMemcpyHtoDAsync', 'hipMemcpyPeer', 'hipMemcpyPeerAsync', 'hipMemcpyToSymbol', 'hipMemcpyToSymbolAsync', 'hipMemcpyWithStream']
    print(f"Extracting copy info for: {str(copyApis)[1:-1]}")

    count = 0
    copy_inserts = [] # rows to bulk insert

    def commitRecords():
        nonlocal copy_inserts
        imp.connection.executemany("insert into rocpd_copyapi(api_ptr_id, stream, size, width, height, kind, src, dst, srcDevice, dstDevice, sync, pinned) values (?,?,?,?,?,?,?,?,?,?,?,?)", copy_inserts)
        imp.connection.commit()
        copy_inserts = []

    for row in connection.execute("select A.api_id, C.string from rocpd_api_ops A join rocpd_api B on B.id = A.api_id join rocpd_string C on C.id = B.args_id where B.apiName_id in (select id from rocpd_string where string in (%s))" % str(copyApis)[1:-1]):
        args = {}
        for line in row[1].split(','):
            key, value = line.partition("=")[::2]
            args[key.strip()] = value.strip()
        stream = args['stream'] if 'stream' in args else ''
        size = args['sizeBytes'] if 'sizeBytes' in args else 0
        width = args['width'] if 'width' in args else 0
        height = args['height'] if 'height' in args else 0
        kind = args['kind'] if 'kind' in args else 0
        src = args['src'] if 'src' in args else ''
        dst = args['dst'] if 'dst' in args else ''
        srcDevice = args['srcDevice'] if 'srcDevice' in args else 0
        dstDevice = args['dstDevice'] if 'dstDevice' in args else 0
        sync = False if 'stream' in args else True
        pinned = False

        copy_inserts.append((row[0], stream, size, width, height, kind, src, dst, srcDevice, dstDevice, sync, pinned))
        if (count % 100000 == 99999):
            commitRecords()
    commitRecords()


#-------------------------------------------------------------------------------
#
#
#
#-------------------------------------------------------------------------------
if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='convert rocprofiler output to an RPD database')
  parser.add_argument('--ops_input_file', type=str, help="hcc_ops_trace.txt from rocprofiler")
  parser.add_argument('--api_input_file', type=str, help="hip_api_trace.txt from rocprofiler")
  parser.add_argument('--hsa_input_file', type=str, help="hsa_api_trace.txt from rocprofiler")
  parser.add_argument('--roctx_input_file', type=str, help="roctx_trace.txt from rocprofiler")
  parser.add_argument('--filter', type=str, help="input filter file")
  parser.add_argument('--input_dir', type=str, help="directory containing rocprofiler intermediate files")
  parser.add_argument('output_rpd', type=str, help="output file")
  args = parser.parse_args()

  if path.exists(args.output_rpd):
      raise Exception(f"Output file: {args.output_rpd} already exists")

  if args.input_dir:
      indir = Path(args.input_dir)
      if (indir/'hcc_ops_trace.txt').exists() and args.ops_input_file == None:
          args.ops_input_file = str(indir/'hcc_ops_trace.txt')
      if (indir/'hip_api_trace.txt').exists() and args.api_input_file == None:
          args.api_input_file = str(indir/'hip_api_trace.txt')
      if (indir/'hsa_api_trace.txt').exists() and args.hsa_input_file == None:
          args.hsa_input_file = str(indir/'hsa_api_trace.txt')
      if (indir/'roctx_trace.txt').exists() and args.roctx_input_file == None:
          args.roctx_input_file = str(indir/'roctx_trace.txt')

  print("Exporting to rpd...")

  connection = sqlite3.connect(args.output_rpd)
  RocpdSchema().writeSchema(connection)

  # Initialize import state
  imp = RocpdImportData();
  imp.initNew(connection);

  if args.ops_input_file:
      print(f"Importing hcc ops from {args.ops_input_file}")
      infile = open(args.ops_input_file, 'r', encoding="utf-8")
      importOps(imp, infile)
      infile.close()

  if args.api_input_file:
      print(f"Importing hip api calls from {args.api_input_file}")
      infile = open(args.api_input_file, 'r', encoding="utf-8")
      importApis(imp, infile)
      infile.close()

  if args.hsa_input_file:
      printf(f"SKIPPING hsa api calls from {args.hsa_input_file}")
      #print(f"Importing hsa api calls from {args.hsa_input_file}")
      #infile = open(args.hsa_input_file, 'r', encoding="utf-8")
      #importHsa(connection, infile)
      #infile.close()

  if args.roctx_input_file:
      print(f"Importing markers from {args.roctx_input_file}")
      infile = open(args.roctx_input_file, 'r', encoding="utf-8")
      importRoctx(imp, infile)
      infile.close()


  populateKernelInfo(imp)
  populateCopyInfo(imp)

  clearApiArgs(imp, "hipGetDevice")
  purgeStrings(imp)

  #connection.execute("vacuum")

  connection.commit()
