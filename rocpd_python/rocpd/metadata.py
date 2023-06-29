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

# Utilities for accessing rocpd_metadata
#

import sys
import os
import argparse
import sqlite3
from rocpd.importer import RocpdImportData


class Metadata:
    def __init__(self, importer):
        self.importer = importer

    def get(self, tag):
        result = self.importer.connection.execute('SELECT value from rocpd_metadata where tag = ?', (tag, ))
        result = result.fetchone()
        return None if result == None else result[0]

    def set(self, tag, value):
        if self.get(tag) == None:
            self.importer.connection.execute('INSERT INTO rocpd_metadata (tag, value) values (?,?)', (tag, value, ))
        else:
            self.importer.connection.execute('UPDATE rocpd_metadata SET value = ? WHERE tag = ?', (value, tag, ))

    def clear(self, tag):
        self.importer.connection.execute('DELETE FROM rocpd_metadata WHERE tag = ?', (tag, ))

    def listAll(self):
        for row in self.importer.connection.execute('SELECT * from rocpd_metadata ORDER BY tag'):
            print(f"{(row[1], row[2])}")
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Utility for reading/writing metadata')
    parser.add_argument('input_rpd', type=str, help="input rpd db")
    parser.add_argument('--list', action='store_true', help="List all metadata rows")
    args = parser.parse_args()

    connection = sqlite3.connect(args.input_rpd)
    importer = RocpdImportData()
    importer.resumeExisting(connection)

    meta = Metadata(importer)
    meta.listAll()
