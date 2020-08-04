import sys
import argparse
import sqlite3
from rocpd.importer import RocpdImportData
from rocpd.deserialize import deserializeApis
from rocpd.subclass import createSubclassTable

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deserialize caching allocator block instrumentation messages and create a subclass table')
    parser.add_argument('input_rpd', type=str, help="input rpd db")
    args = parser.parse_args()


    connection = sqlite3.connect(args.input_rpd)
    importData = RocpdImportData()
    importData.resumeExisting(connection) # load the current db state

    roctxApis = ["UserMarker"]

    blockApis = [
        'BlockAlloc',
        'BlockFreeDeallocate',
        'BlockInsertEvents',
        'ProcessEvents',
        'BlockFreeDeactivate',
    ]

    argTypes = {
       'size' : 'integer NOT NULL',
       'block' : 'varchar(18) NOT NULL',
       'stream' : 'integer NOT NULL',
       'event' : 'varchar(18) NOT NULL',
    }

    print(f"Deserializing apis in: {str(roctxApis)[1:-1]}")
    deserializeApis(importData, roctxApis)
    print(f"Creating subclass table for 'block' apis: {blockApis}")
    createSubclassTable(importData, 'api', 'block', blockApis, argTypes)

