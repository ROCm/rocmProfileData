###########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
###########################################################################

from setuptools import setup, find_packages

setup(name = 'rocpd',
      version = '1.0',
      description = 'RocmProfileData profiling format',
      #packages = find_packages(),
      packages = { 'rocpd' },
      include_package_data=True, 
      python_requires='>=3.6',
      zip_safe=False,
)
