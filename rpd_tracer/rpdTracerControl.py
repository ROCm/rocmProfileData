
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

from ctypes import CDLL
from ctypes.util import find_library
import platform
import multiprocessing
import threading
import os
import sys
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
    __active = True

    @classmethod
    def loadLibrary(cls):
        if cls.__rpd == None and cls.__active == True:
            os.environ["RPDT_AUTOSTART"] = "0"
            cls.__rpd = CDLL(find_library("rpd_tracer"))

    @classmethod
    def initializeFile(cls):
        # Only the top parent process will initialize the trace file
        if isChildProcess() or os.getenv("RPDT_FILENAME"):
            cls.__initFile = False
        if cls.__initFile == True and cls.__active == True:
            if os.path.exists(cls.__filename):
                os.remove(cls.__filename)
            # Create new file and write schema
            schema = RocpdSchema()
            connection = sqlite3.connect(cls.__filename)
            schema.writeSchema(connection)
            connection.commit()
            connection.close()
            #
            #os.environ["RPDT_FILENAME"] = cls.__filename
            cls.__initFile = False   
        os.environ["RPDT_FILENAME"] = cls.__filename

    # You can set the output filename and optionally append to an exiting file.
    #   This must be done on the main process before any class instances are created.

    # When using python multiprocessing, you must crate an rpdTracerControl instance on the
    #   main process before spawning processes.  This initializes the output file and sets
    #   up the envirnoment that the child processes will inherit

    @classmethod
    def setFilename(cls, name, append = False):
        if os.getenv("RPDT_FILENAME"):
            cls.__filename = os.getenv("RPDT_FILENAME")
            if name != cls.__filename:
                raise Warning(f"RPDT_FILENAME is already initialized. Logging into {cls.__filename}")
            return

        if cls.__rpd != None:
            raise RuntimeError("Trace file name can not be changed once logging")
        if isChildProcess():
            raise RuntimeError("Trace file name can not be changed by sub-processes")


        cls.__filename = name
        #os.environ["RPDT_FILENAME"] = cls.__filename
        if append:
            cls.__initFile = False

    @classmethod
    def skipCreate(cls):
        cls.__initFile = False

    @classmethod
    def skipLoad(cls):
       # If the tracer is not already loaded, then everything is a no-op.
       if os.environ.get("RPDT_LOADED") == None:
           cls.__active = False

    def __init__(self):
        rpdTracerControl.initializeFile()
        rpdTracerControl.loadLibrary()

    def __del__(self):
        pass

    def start(self):
        if rpdTracerControl.__rpd:
            rpdTracerControl.__rpd.rpdstart()

    def stop(self):
        if rpdTracerControl.__rpd:
            rpdTracerControl.__rpd.rpdstop()

    def flush(self):
        if rpdTracerControl.__rpd:
            rpdTracerControl.__rpd.rpdflush()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def rangePush(self, domain: str, apiName: str, args: str):
        if rpdTracerControl.__rpd:
            rpdTracerControl.__rpd.rpd_rangePush(bytes(domain, encoding='utf-8'), bytes(apiName, encoding='utf-8'), bytes(args, encoding='utf-8'))

    def rangePop(self):
        if rpdTracerControl.__rpd:
            rpdTracerControl.__rpd.rpd_rangePop()


    # python stack tracing

    def __trace_callback(self, frame, event, arg):
        if frame.f_code.co_name.startswith("__") or frame.f_code.co_name == "rangePush" or frame.f_code.co_name == "rangePop":
            return None
        if event == 'call':
            self.rangePush("python", frame.f_code.co_name, f"{frame.f_code.co_filename}:{frame.f_code.co_firstlineno}");
        if event == 'return':
            self.rangePop()

    def setPythonTrace(self, doTrace: bool):
        if doTrace:
            sys.setprofile(self.__trace_callback)
            try:
                threading.setprofile_all_threads(self.__trace_callback)	#python 3.12+
            except:
                threading.setprofile(self.__trace_callback)
                # FIXME: handle running threads
        else:
            sys.setprofile(None)
            try:
                threading.setprofile_all_threads(None)	#python 3.12+
            except:
                pass
                threading.setprofile(None)
                # FIXME: handle running threads

