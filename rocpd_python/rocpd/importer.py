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
# Utility classes to simplify generating rpd files
#
#



class RocpdImportData:
    def __init__(self):
        #Set up primary keys
        self.string_id = 1
        self.op_id = 1
        self.api_id = 1
        #self.hsa_id = 1
        # Dicts
        self.strings = {}    # string -> id
        # Empty string
        self.empty_string_id = 1
        self.connection = None
        self.string_inserts = []

    def __del__(self):
        self.commitStrings(True)
        pass

# initialize 
    def initNew(self, connection):
        self.connection = connection
        self.initEmptyString()

    def resumeExisting(self, connection):
        self.connection = connection
        self.buildStringCache()
        self.buildCurrentIds()

    def initEmptyString(self):
        self.empty_string_id = self.string_id
        self.string_id = self.string_id + 1
        self.strings[""] = self.empty_string_id
        self.connection.execute("insert into rocpd_string(id, string) values (?,?)", (self.empty_string_id, ""))

    def buildStringCache(self):
        self.strings = {}
        # Find the empty string, create if needed
        self.empty_string_id = -1
        for row in self.connection.execute("select id from rocpd_string where string=''"):
            self.empty_string_id = row[0]
        if self.empty_string_id == -1:
            self.initEmptyString()

        for row in self.connection.execute("select id from rocpd_string order by id desc limit 1"):
            self.string_id = row[0] + 1

        for row in self.connection.execute("select id, string from rocpd_string"):
            self.strings[row[1]] = row[0]

    def buildCurrentIds(self):
        for row in self.connection.execute("select id from rocpd_op order by id desc limit 1"):
            self.op_id = row[0] + 1
        for row in self.connection.execute("select id from rocpd_api order by id desc limit 1"):
            self.api_id = row[0] + 1


# Handle string cache and string table insert in one place

    def getStringId(self, val):
        id = None
        try:
            id = self.strings[val]
        except:
            self.strings[val] = self.string_id
            self.string_inserts.append((self.string_id, val))
            id = self.string_id
            self.string_id = self.string_id + 1
        return id

    def commitStrings(self, commit = False):
        #if self.string_inserts.count() > 0:
        if len(self.string_inserts) > 0:
            self.connection.executemany("insert into rocpd_string(id, string) values (?,?)", self.string_inserts)
            if commit == True:
                self.connection.commit()
            self.string_inserts = []



