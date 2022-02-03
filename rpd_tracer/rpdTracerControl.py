###########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
###########################################################################

from ctypes import CDLL
from ctypes.util import find_library
import os
import sqlite3
from rocpd.schema import RocpdSchema

class rpdTracerControl:
    filename = "trace.rpd"
    __rpd = None    # the dll
    __initFile = True

    @classmethod
    def loadLibrary(cls):
        if cls.__rpd == None:
            os.environ["RPDT_AUTOSTART"] = "0"
            cls.__rpd = CDLL(find_library("rpd_tracer"))

    @classmethod
    def initializeFile(cls):
        if cls.__initFile == True:
            if os.path.exists(cls.filename):
                os.remove(cls.filename)
            # Create new file and write schema
            schema = RocpdSchema()
            connection = sqlite3.connect(cls.filename)
            schema.writeSchema(connection)
            connection.commit()
            connection.close()
            #
            os.environ["RPDT_FILENAME"] = cls.filename
            cls.__initFile = False   

    # Multiple concurrent processes will each reinitialize the output file
    #   workaround: Call rpdTracerControl.skipFileInit() from "spawned" processes

    @classmethod
    def skipFileInit(cls):
        cls.__initFile = False

    def __init__(self):
        rpdTracerControl.initializeFile()
        rpdTracerControl.loadLibrary()

    def __del__(self):
        pass

    def start(self):
        rpdTracerControl.__rpd.rpdstart()

    def stop(self):
        rpdTracerControl.__rpd.rpdstop()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
