###########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
###########################################################################

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
        # Find the emptry string, create if needed
        self.empty_string_id = -1
        for row in self.connection.execute("select id from rocpd_string where string=''"):
            self.empty_string_id = row[0]
        if self.empty_string_id == -1:
            self.initEmptyString()

        for row in self.connection.execute("select id from rocpd_string order by id desc limit 1"):
            self.string_id = row[0] + 1

        for row in self.connection.execute("select id, string from rocpd_string"):
            self.strings[row[0]] = row[1]

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



