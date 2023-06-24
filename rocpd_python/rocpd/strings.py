###########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
###########################################################################

# Create and auxilarly table to express parent/child api calls
#
#
#

import argparse
import sqlite3
from rocpd.importer import RocpdImportData


def cleanStrings(imp, fix_autograd):
    # Create backup table
    imp.connection.execute('CREATE TEMPORARY TABLE IF NOT EXISTS "rocpd_string_original" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "string" varchar(4096) NOT NULL)')
    imp.connection.execute('INSERT into rocpd_string_original SELECT * from rocpd_string')


    # Normalize autograd strings
    if fix_autograd:
        imp.connection.execute("""
            UPDATE rocpd_string_original set string = SUBSTR(string, 1, INSTR(string, ", seq") - 1) where string like "%, seq%"
        """)
        imp.connection.execute("""
            UPDATE rocpd_string_original set string = SUBSTR(string, 1, INSTR(string, ", op_id") - 1) where string like "%, op_id%"
        """)
        imp.connection.execute("""
            UPDATE rocpd_string_original set string = SUBSTR(string, 1, INSTR(string, ", sizes") - 1) where string like "%, sizes%"
        """)
        imp.connection.execute("""
            UPDATE rocpd_string_original set string = SUBSTR(string, 1, INSTR(string, ", input_op_ids") - 1) where string like "%, input_op_ids%"
        """)



    # Drop, recreate, and populate the string table
    imp.connection.execute("""
        DROP TABLE "rocpd_string";
        """)
    imp.connection.execute("""
        CREATE TABLE IF NOT EXISTS "rocpd_string" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "string" varchar(4096) NOT NULL);
        """)
    imp.connection.execute("""
        CREATE INDEX "rocpd_strin_string_c7b9cd_idx" ON "rocpd_string" ("string");
        """)
    imp.connection.execute("""
        INSERT into rocpd_string(string) SELECT distinct string from rocpd_string_original order by id;
        """)

    # Map from old id to new; UPDATE all table with new string id
    imp.connection.execute("""
        create temporary view mapper as SELECT A.id as before, B.id as after from rocpd_string_original A join rocpd_string B on B.string = A.string;
        """)

    imp.connection.execute("""
        UPDATE rocpd_api set apiName_id = (SELECT after from mapper A where apiName_id=A.before);
        """)
    imp.connection.execute("""
        UPDATE rocpd_api set args_id = (SELECT after from mapper A where args_id=A.before);
        """)
    imp.connection.execute("""
        UPDATE rocpd_op set opType_id = (SELECT after from mapper A where opType_id=A.before);
        """)
    imp.connection.execute("""
        UPDATE rocpd_op set description_id = (SELECT after from mapper A where description_id=A.before);
        """)
    imp.connection.execute("""
        UPDATE rocpd_kernelapi set kernelName_id = (SELECT after from mapper A where kernelName_id=A.before);
        """)

    imp.connection.commit()
    imp.resumeExisting(imp.connection)	# reload state

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Utilities for tidying rocpd_string table')
    parser.add_argument('input_rpd', type=str, help="input rpd db")
    parser.add_argument('--dedupe', action='store_true', help="Remove duplicate strings")
    parser.add_argument('--clean_autograd', action='store_true', help="Remove 'op' and 'seq' tags from strings")
    args = parser.parse_args()

    connection = sqlite3.connect(args.input_rpd)

    importData = RocpdImportData()
    importData.resumeExisting(connection) # load the current db state

    if args.dedupe or args.clean_autograd:
        cleanStrings(importData, args.clean_autograd)
    else:
        print("No action taken.  Check --help")

