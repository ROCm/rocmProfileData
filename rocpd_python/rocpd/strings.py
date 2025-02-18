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


    # Make a list of all columns that reference rocpd_string
    #   These will need to be updated and relinked
    #   Also, we can detect and remove unreferences strings

    # Well known references
    string_users = [
    ("rocpd_op", "description_id"),
    ("rocpd_op", "opType_id"),
    ("rocpd_api", "apiName_id"),
    ("rocpd_api", "args_id"),
    ("rocpd_kernelapi", "kernelName_id"),
    ]
    # Explicity declared in rocpd_metadata.   Format is: tag = 'references::rocpd_string.id', value = '("table_name", "column_name")'
    for row in imp.connection.execute("SELECT value from rocpd_metadata where tag='references::rocpd_string.id'"):
        value = eval(row[0])
        if type(value) == tuple:
            if value not in string_users:
                string_users.append(value)

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


    # Clean up unrefferenced strings.  Tools can change strings (names, args, etc) and may leave strings that are no longer being used.
    imp.connection.execute("""
        CREATE TEMPORARY TABLE IF NOT EXISTS "activeString" ("id" integer NOT NULL PRIMARY KEY);
        """)
    for column in string_users:
        query = f"""INSERT OR IGNORE INTO "activeString" SELECT {column[1]} from {column[0]}"""
        #print(query)
        imp.connection.execute(query)


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
        INSERT into rocpd_string(string) SELECT distinct string from rocpd_string_original where id in (SELECT id FROM "activeString") order by id;
        """)


    # Map from old id to new; UPDATE all table with new string id
    # WARNING: the 2nd union term handles a string referenced but not present in rocpd_string.
    #    E.g. a corrupt file.
    imp.connection.execute("""
        CREATE TEMPORARY VIEW IF NOT EXISTS mapper as SELECT A.id as before, B.id as after from rocpd_string_original A join rocpd_string B on B.string = A.string UNION ALL select id, 1 from activeString where id not in (select distinct id from rocpd_string_original);
        """)

    for column in string_users:
        query = f"""UPDATE {column[0]} set {column[1]} = (SELECT after from mapper A where {column[1]}=A.before) """
        #print(query)
        imp.connection.execute(query)

    # cleanup
    imp.connection.execute("""
        DROP TABLE rocpd_string_original
        """)
    imp.connection.execute("""
        DROP TABLE "activeString"
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

