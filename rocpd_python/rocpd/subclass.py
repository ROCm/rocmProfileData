###########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
###########################################################################

# Create subclass table for a set of apis/ops exposing args as columns
#
# input1: parent class ('api' || 'op')
# input2: name of subclass
# input3: list of apis/ops names to include
#
# New columns/fields will be read out of args (apis) or description (op)
# argument format is: arg1 = value1, arg2 = value2, ...
#
#

import sys
import os
import re
import argparse
import sqlite3
from rocpd.importer import RocpdImportData


def createSubclassTable(importData, baseClass, subClass, events, argTypes):
    if baseClass != 'api' and baseClass != 'op':
        raise("baseClass must be 'api' or 'op'")
    queryString = \
        'select distinct B.string from rocpd_api A join rocpd_string B on B.id = A.args_id where A.apiName_id in\n\
            (select id from rocpd_string where string in (%s))' \
        if baseClass == 'api' else \
        'select distinct B.string from rocpd_op A join rocpd_string B on B.id = A.description_id where A.opType_id in\n\
            (select id from rocpd_string where string in (%s))'


    # Scan all events of subclass.  Find superset of args 
    args = {}
    for row in importData.connection.execute(queryString % str(events)[1:-1]):
        for line in row[0].split('|'):
            key,value = line.partition("=")[::2]
            args[key.strip()] = True
    try:
        args.pop('s_api')
    except KeyError: pass
    try:
        args.pop('s_op')
    except KeyError: pass


    # Create the subclass table
    elements = []
    queryString = f'create table if not exists "rocpd_{subClass}{baseClass}" ('
    if baseClass == 'api':
        queryString = queryString + '"api_ptr_id" integer NOT NULL PRIMARY KEY REFERENCES "rocpd_api" ("id") DEFERRABLE INITIALLY DEFERRED, '
    else:	# baseClass == 'op'
        queryString = queryString + '"op_ptr_id" integer NOT NULL PRIMARY KEY REFERENCES "rocpd_op" ("id") DEFERRABLE INITIALLY DEFERRED, '
    
    for arg in args:
        queryString = queryString + f'"{arg}" '
        if arg in argTypes:
            queryString = queryString + argTypes[arg] + ', '
        else:
            queryString = queryString + 'varchar(255), '
    queryString = queryString.strip(', ')
    queryString = queryString + ')'
    importData.connection.execute(queryString)


    #
    # Populate the subclass table
    #
    count = 0
    inserts = [] # rows to bulk insert
    def commitRecords():
        nonlocal inserts
        nonlocal insertQueryString
        importData.connection.executemany(insertQueryString, inserts)
        importData.connection.commit()
        inserts = []

    # build the insert string
    insertQueryString = f'insert into rocpd_{subClass}{baseClass}('
    if baseClass == 'api':
        insertQueryString = insertQueryString + 'api_ptr_id,'
    else:   # baseClass == 'op'
        insertQueryString = insertQueryString + 'op_ptr_id,'
    for arg in args:
        insertQueryString = insertQueryString + f'{arg},'
    insertQueryString = insertQueryString.strip(',') + ') values (?,'
    for arg in args:
        insertQueryString = insertQueryString + '?,'
    insertQueryString = insertQueryString.strip(',') + ')'

    # Fetch the rows to generate subclass rows
    queryString = \
        'select A.id, B.string from rocpd_api A join rocpd_string B on B.id = A.args_id where A.apiName_id in\n\
            (select id from rocpd_string where string in (%s))' \
        if baseClass == 'api' else \
        'select A.id, B.string from rocpd_op A join rocpd_string B on B.id = A.description_id where A.opType_id in\n\
            (select id from rocpd_string where string in (%s))'

    for row in importData.connection.execute(queryString % str(events)[1:-1]):
        argvals = {}
        for line in row[1].split('|'):
            key,value = line.partition("=")[::2]
            argvals[key.strip()] = value.strip()
        val_list = []
        val_list.append(row[0])		# base class ptr_id
        for arg in args:			# each subbclass field
            if arg in argvals:
                val_list.append(argvals[arg])
            else:
                val_list.append('')
        inserts.append(tuple(val_list))
        count = count + 1
        if (count % 100000 == 99999):
            commitRecords()
    commitRecords()

    # Create a subclass view
    if baseClass == 'api':
        importData.connection.execute(f"create view {subClass} as select * from api A join rocpd_{subClass}{baseClass} B on B.api_ptr_id=A.id") 
    else: # baseClass == 'op'
        importData.connection.execute(f"create view {subClass} as select * from op A join rocpd_{subClass}{baseClass} B on B.op_ptr_id=A.id") 
    importData.connection.commit()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create subclass table for a set of apis/ops exposing args as columns.\nThis should be run interactively by calling createSubclassTable() with proper arguments.\nCheck the example folder.')
    args = parser.parse_args()

