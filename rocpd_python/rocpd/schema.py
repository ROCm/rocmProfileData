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
