###########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
###########################################################################

from distutils.core import setup, Extension

setup (name = 'rpdTracer',
       version = '1.0',
       description = 'Tracer control from user code',
       py_modules = ['rpdTracerControl'],
       )