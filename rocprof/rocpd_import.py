# Create an rpd file from rocprofiler output files

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


def createSchema(connection):
    connection.execute('CREATE TABLE IF NOT EXISTS "rocpd_string" ("id" integer NOT NULL PRIMARY KEY, "string" varchar(4096) NOT NULL)')
    connection.execute('CREATE TABLE IF NOT EXISTS "rocpd_op" ("id" integer NOT NULL PRIMARY KEY, "gpuId" integer NOT NULL, "queueId" integer NOT NULL, "sequenceId" integer NOT NULL, "completionSignal" varchar(18) NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "description_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "opType_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED)')
    connection.execute('CREATE TABLE IF NOT EXISTS "rocpd_api" ("id" integer NOT NULL PRIMARY KEY, "pid" integer NOT NULL, "tid" integer NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "apiName_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "args_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED)')
    connection.execute('CREATE TABLE IF NOT EXISTS "rocpd_api_ops" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "api_id" integer NOT NULL REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, "op_id" integer NOT NULL REFERENCES "rocpd_op" ("id") DEFERRABLE INITIALLY DEFERRED)')
    connection.execute('CREATE TABLE IF NOT EXISTS "rocpd_hsaApi" ("id" integer NOT NULL PRIMARY KEY, "pid" integer NOT NULL, "tid" integer NOT NULL, "start" integer NOT NULL, "end" integer NOT NULL, "apiName_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "args_id" integer NOT NULL REFERENCES "rocpd_string" ("id") DEFERRABLE INITIALLY DEFERRED, "return" integer NOT NULL)')
    connection.commit()


def createViews(connection):
    connection.execute("CREATE VIEW api AS SELECT rocpd_api.id,pid,tid,start,end,A.string AS apiName, B.string AS args FROM rocpd_api INNER JOIN rocpd_string A ON A.id = rocpd_api.apiName_id INNER JOIN rocpd_string B ON B.id = rocpd_api.args_id")
    connection.execute("CREATE VIEW op AS SELECT rocpd_op.id,gpuId,queueId,sequenceId,start,end,A.string AS description, B.string AS opType FROM rocpd_op INNER JOIN rocpd_string A ON A.id = rocpd_op.description_id INNER JOIN rocpd_string B ON B.id = rocpd_op.opType_id")
    connection.execute("CREATE VIEW top AS SELECT A.string as KernelName, count(A.string) as TotalCalls, sum(rocpd_op.end-rocpd_op.start) / 1000 as TotalDuration, (sum(rocpd_op.end-rocpd_op.start)/count(A.string)) / 1000 as Ave, sum(rocpd_op.end-rocpd_op.start) * 100.0 / (select sum(end-start) from rocpd_op) as Percentage FROM rocpd_api_ops INNER JOIN rocpd_op ON rocpd_api_ops.op_id = rocpd_op.id INNER JOIN rocpd_string A ON A.id = rocpd_op.description_id group by KernelName order by TotalDuration desc")
    connection.execute("CREATE VIEW busy AS select A.gpuId, GpuTime, WallTime, GpuTime*1.0/WallTime as Busy from (select gpuId, sum(end-start) as GpuTime from rocpd_op group by gpuId) A INNER JOIN (select max(end) - min(start) as WallTime from rocpd_op)")
    connection.commit()


def initEmptyString(connection):
  global empty_string_id
  global string_id
  global strings
  empty_string_id = string_id
  string_id = string_id + 1
  strings[""] = empty_string_id
  connection.execute("insert into rocpd_string(id, string) values (?,?)", (empty_string_id, ""))


def importOps(connection, infile):
    exp = re.compile("^(\d*):(\d*)\s+(\d*):(\d*)\s+(\w+):(\d*).*$")
    count = 0;
    op_inserts = [] # rows to bulk insert
    string_inserts = [] # rows to bulk insert
    api_ops_inserts = [] # rows to bulk insert

    global strings
    global string_id 
    global op_id 
    #global api_id 
    #global hsa_id 

    def commitRecords():
        nonlocal op_inserts
        nonlocal string_inserts
        nonlocal api_ops_inserts
        #print(count+1)
        #print("--------------------------------------------------------------------------")
        #print(string_inserts)
        #print("++++")
        #print(op_inserts)
        #print("####")
        #print(api_ops_inserts)
        connection.executemany("insert into rocpd_string(id, string) values (?,?)", string_inserts)
        connection.executemany("insert into rocpd_op(id, gpuId, queueId, sequenceId, completionSignal,  start, end, description_id, opType_id) values (?,?,?,'','',?,?,?,?)", op_inserts)
        connection.executemany("insert into rocpd_api_ops(api_id, op_id) values (?,?)", api_ops_inserts)
        connection.commit()
        op_inserts = []
        string_inserts = []
        api_ops_inserts = []

    for line in infile:
        m = exp.match(line)
        if m:
            try:
                name = strings[m.group(5)]
                #print(f"   : {m.group(5)} {name}")
            except:
                strings[m.group(5)] = string_id
                string_inserts.append((string_id, m.group(5)))
                #print(f"+++: {m.group(5)} {string_id}")
                name = string_id
                string_id = string_id + 1
            try:
                desc = strings[""]
                #print(f"   : {m.group(6)} {desc}")
            except:
                strings[""] = string_id
                string_inserts.append((string_id, ""))
                #print(f"+++: {m.group(6)} {string_id}")
                desc = string_id
                string_id = string_id + 1

            if len(m.group(6)) > 0:
                api_ops_inserts.append((int(m.group(6)), op_id))

            op_inserts.append((op_id, m.group(3), m.group(4), m.group(1), m.group(2), desc, name))
            op_id = op_id + 1
            count = count + 1
        if (count % 100000 == 99999):
            commitRecords()
    commitRecords()


def importApis(connection, infile):
    exp =    re.compile("^(\d*):(\d*)\s+(\d*):(\d*)\s+(\w+)\((.*)\).*$")
    expfix = re.compile("^(\d*):(\d*)\s+(\d*):(\d*)\s+(\w+)\((.*)\)( kernel=.*)$")
    count = 0
    api_inserts = [] # rows to bulk insert
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

    for line in infile:
        kernstring = None
        m = expfix.match(line)
        if m:
            kernstring = m.group(6) + ', ' + m.group(7)
        else:
            m = exp.match(line)
        if m:
            try:
                api = strings[m.group(5)]
            except:
                strings[m.group(5)] = string_id
                string_inserts.append((string_id, m.group(5)))
                api = string_id
                string_id = string_id + 1
            kernstring = kernstring or m.group(6)
            try:
                arg = strings[kernstring]
            except:
                strings[kernstring] = string_id
                string_inserts.append((string_id, kernstring))
                arg = string_id
                string_id = string_id + 1

            api_inserts.append((api_id, m.group(3), m.group(4), m.group(1), m.group(2), api, arg))
            api_id = api_id + 1
            count = count + 1
        if (count % 100000 == 99999):
            commitRecords()
    commitRecords()


def importHsa(connection, infile):
    exp = re.compile("^(\d*):(\d*)\s+(\d*):(\d*)\s+(\w+)\((.*)\)\s*=\s*(\d*).*$")
    count = 0
    hsa_inserts = [] # rows to bulk insert
    string_inserts = [] # rows to bulk insert

    global strings
    global string_id

    def commitRecords():
        nonlocal hsa_inserts
        nonlocal string_inserts

def importRoctx(connection, infile):
    exp = re.compile("^(\d*)\s+(\d*):(\d*)\s+(\d+):\"(.*)\".*$")
    count = 0;
    stack = []
    api_inserts = [] # rows to bulk insert
    string_inserts = [] # rows to bulk insert

    global strings
    global string_id

    def commitRecords():
        nonlocal api_inserts
        nonlocal string_inserts


# Clear args for a given api
def clearApiArgs(connection, apiname):
    global empty_string_id
    connection.execute("update rocpd_api set args_id = ? where apiName_id in (select id from rocpd_string where string = ?)", (empty_string_id, apiname))
    connection.commit()


# Remove unreferenced rows from string table
def purgeStrings(connection):
    connection.execute("delete from rocpd_string where id not in (\
            select distinct apiName_id from rocpd_api \
            union all select distinct args_id from rocpd_api \
            union all select distinct description_id from rocpd_op \
            union all select distinct opType_id from rocpd_op \
            union all select distinct kernelName_id from rocpd_kernelop \
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

def populateKernelInfo(connection):
    kernelApis = ['hipHccModuleLaunchKernel', 'hipLaunchKernel', 'hipExtModuleLaunchKernel']
    print(f"Extracting kernel info for: {str(kernelApis)[1:-1]}")

    count = 0
    kernel_inserts = [] # rows to bulk insert
    string_inserts = [] # rows to bulk insert

    global strings
    global string_id

    def commitRecords():
        nonlocal kernel_inserts
        nonlocal string_inserts
        connection.executemany("insert into rocpd_string(id, string) values (?,?)", string_inserts)
        connection.executemany("insert into rocpd_kernelop(op_ptr_id, gridX, gridY, gridZ, workgroupX, workgroupY, workgroupZ, groupSegementSize, privateSegementSize, codeObject_id, kernelName_id, kernelArgAddress, aquireFence, releaseFence) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", kernel_inserts)
        connection.commit()
        kernel_inserts = []
        string_inserts = []


    for row in connection.execute("select A.op_id, C.string from rocpd_api_ops A join rocpd_api B on B.id = A.api_id join rocpd_string C on C.id = B.args_id where B.apiName_id in (select id from rocpd_string where string in (%s))" % str(kernelApis)[1:-1]):
        args = {}
        for line in row[1].split(','):
            key, value = line.partition("=")[::2]
            args[key.strip()] = value.strip()
        gridx = args['gridDimX'] if 'gridDimX' in args else 0
        gridy = args['gridDimY'] if 'gridDimY' in args else 0
        gridz = args['gridDimZ'] if 'gridDimZ' in args else 0
        bdimx = args['blockDimX'] if 'blockDimX' in args else 0
        bdimy = args['blockDimY'] if 'blockDimY' in args else 0
        bdimz = args['blockDimZ'] if 'blockDimZ' in args else 0
        shmem = args['sharedMemBytes'] if 'sharedMemBytes' in args else 0
        prmem = 0
        kernstring = args['kernel'] if 'kernel' in args else ''
        kargs = args['args'] if 'args' in args else ''
        aqfence = '';
        relfence = '';
        try:
            kernel = strings[kernstring]
        except:
            strings[kernstring] = string_id
            string_inserts.append((string_id, kernstring))
            kernel = string_id
            string_id = string_id + 1

        kernel_inserts.append((row[0], gridx, gridy, gridz, bdimx, bdimy, bdimz, shmem, prmem, 0, kernel, kargs, aqfence, relfence))
        count = count + 1
        if (count % 100000 == 99999):
            commitRecords()
    commitRecords()
    # copy the kernel names into the base op table's description
    # Most use cases will just want the kernel name and can avoid joining KernelOp
    connection.execute("update rocpd_op set description_id = (select kernelName_id from rocpd_kernelop A join rocpd_op B on B.id = A.op_ptr_id where rocpd_op.id = B.id) where rocpd_op.id in (select op_ptr_id from rocpd_kernelop)")
    connection.commit()


# Build op child classes for the following apis
# hipMemcpyWithStream
# hipMemsetAsync
# hipMemcpyAsync

def populateCopyInfo(connection):
    copyApis = ['hipMemcpyWithStream', 'hipMemsetAsync', 'hipMemcpyAsync']
    print(f"Extracting copy info for: {str(copyApis)[1:-1]}")

    count = 0
    copy_inserts = [] # rows to bulk insert

    global strings
    global string_id

    def commitRecords():
        nonlocal copy_inserts
        connection.executemany("insert into rocpd_copyop(op_ptr_id, size, src, dst, sync, pinned) values (?,?,?,?,?,?)", copy_inserts)
        connection.commit()
        copy_inserts = []

    for row in connection.execute("select A.op_id, C.string from rocpd_api_ops A join rocpd_api B on B.id = A.api_id join rocpd_string C on C.id = B.args_id where B.apiName_id in (select id from rocpd_string where string in (%s))" % str(copyApis)[1:-1]):
        args = {}
        for line in row[1].split(','):
            key, value = line.partition("=")[::2]
            args[key.strip()] = value.strip()
        size = args['sizeBytes'] if 'sizeBytes' in args else 0
        src = -1
        dst = -1
        sync = False
        pinned = False

        copy_inserts.append((row[0], size, src, dst, sync, pinned))
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
  #createSchema(connection)
  #createViews(connection)
  RocpdSchema().writeSchema(connection)

  #Set up primary keys
  string_id = 1
  op_id = 1
  api_id = 1
  hsa_id = 1

  # Dicts
  strings = {}    # string -> id

  # Empty string
  empty_string_id = 1
  initEmptyString(connection)

  if args.ops_input_file:
      print(f"Importing hcc ops from {args.ops_input_file}")
      infile = open(args.ops_input_file, 'r', encoding="utf-8")
      importOps(connection, infile)
      infile.close()

  if args.api_input_file:
      print(f"Importing hip api calls from {args.api_input_file}")
      infile = open(args.api_input_file, 'r', encoding="utf-8")
      importApis(connection, infile)
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
      importRoctx(connection, infile)
      infile.close()


  populateKernelInfo(connection)
  populateCopyInfo(connection)

  clearApiArgs(connection, "hipGetDevice")
  purgeStrings(connection)

  #connection.execute("vacuum")

  connection.commit()
