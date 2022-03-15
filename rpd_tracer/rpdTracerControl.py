###########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
###########################################################################

from ctypes import CDLL
from ctypes.util import find_library
import platform
import multiprocessing
import os
import sqlite3
from rocpd.schema import RocpdSchema

def isChildProcess() -> bool:
    version = platform.python_version_tuple()
    if int(version[0]) >=3 and int(version[1]) >= 8:
        return multiprocessing.parent_process() != None
    else:
        return type(multiprocessing.current_process()) == multiprocessing.Process

class rpdTracerControl:
    __filename = "trace.rpd"
    __rpd = None    # the dll/
    __initFile = True

    @classmethod
    def loadLibrary(cls):
        if cls.__rpd == None:
            os.environ["RPDT_AUTOSTART"] = "0"
            cls.__rpd = CDLL(find_library("rpd_tracer"))

    @classmethod
    def initializeFile(cls):
        # Only the top parent process will initialize the trace file
        if isChildProcess():
            cls.__initFile = False

        if cls.__initFile == True:
            if os.path.exists(cls.__filename):
                os.remove(cls.__filename)
            # Create new file and write schema
            schema = RocpdSchema()
            connection = sqlite3.connect(cls.__filename)
            schema.writeSchema(connection)
            connection.commit()
            connection.close()
            #
            os.environ["RPDT_FILENAME"] = cls.__filename
            cls.__initFile = False   


    # You can set the output filename and optionally append to an exiting file.
    #   This must be done on the main process before any class instances are created.

    # When using python multiprocessing, you must crate an rpdTracerControl instance on the
    #   main process before spawning processes.  This initializes the output file and sets
    #   up the envirnoment that the child processes will inherit

    @classmethod
    def setFilename(cls, name, append = False):
        if cls.__rpd != None:
            raise RuntimeError("Trace file name can not be changed once logging")
        if isChildProcess():
            raise RuntimeError("Trace file name can not be changed by sub-processes")

        cls.__filename = name
        if append:
            os.environ["RPDT_FILENAME"] = cls.__filename
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
