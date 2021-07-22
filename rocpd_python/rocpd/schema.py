#
#   Utility Class to create the rocpd schema on an existing sqlite connection
#     
#       Requires a current copy of the schema in the 'schema' subdirectory
#       Executes the contained sql 'scripts' to create the schema
#

import os
import sqlite3
import argparse
from pathlib import Path

class RocpdSchema:

    def __init__(self):
        schemadir = Path(os.path.dirname(os.path.abspath(__file__)))

        with open(str(schemadir/'schema_data/tableSchema.cmd'), 'r') as schema:
            self.tableSchema = schema.read()
        with open(str(schemadir/'schema_data/indexSchema.cmd'), 'r') as schema:
            self.indexSchema = schema.read()
        with open(str(schemadir/'schema_data/index2Schema.cmd'), 'r') as schema:
            self.index2Schema = schema.read()
        with open(str(schemadir/'schema_data/utilitySchema.cmd'), 'r') as schema:
            self.utilitySchema = schema.read()

    def writeSchema(self, connection):
        connection.executescript(self.tableSchema)
        connection.executescript(self.indexSchema)
        connection.executescript(self.utilitySchema)
 
    def writeBasicSchema(self, connection):
        connection.executescript(self.tableSchema)
        connection.executescript(self.indexSchema)

    def writeFullSchema(self, connection):
        connection.executescript(self.tableSchema)
        connection.executescript(self.indexSchema)
        connection.executescript(self.index2Schema)
        connection.executescript(self.utilitySchema)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert rocprofiler output to an RPD database')
    parser.add_argument('--create', type=str, help="filename in create empty db")
    args = parser.parse_args()

    schema = RocpdSchema()

    if args.create:
        print(f"Creating empty rpd: {args.create}")
        connection = sqlite3.connect(args.create)
        schema.writeSchema(connection)
        connection.commit()
    else:
        print(schema.tableSchema)
