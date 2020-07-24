#
#   Utility Class to create the rocpd schema on an existing sqlite connection
#     
#       Requires a current copy of the schema in the 'schema' subdirectory
#       Executes the contained sql 'scripts' to create the schema
#

import os
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
    schema = RocpdSchema()
    print(schema.tableSchema)
