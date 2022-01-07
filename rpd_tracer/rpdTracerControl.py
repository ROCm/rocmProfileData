from ctypes import CDLL
from ctypes.util import find_library
import os

class rpdTracerControl:
    rpd = None

    @classmethod
    def loadLibrary(cls):
        if cls.rpd == None:
            os.environ["RPDT_AUTOSTART"] = "0"
            #cls.rpd = CDLL("./rpd_tracer.so")
            cls.rpd = CDLL(find_library("rpd_tracer"))

    def __init__(self):
        rpdTracerControl.loadLibrary()

    def __del__(self):
        pass

    def start(self):
        rpdTracerControl.rpd.rpdstart()

    def stop(self):
        rpdTracerControl.rpd.rpdstop()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
