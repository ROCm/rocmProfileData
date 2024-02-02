from distutils.core import setup, Extension

#module1 = Extension('hipMarker',
#                    sources = ['hipMarkerModule.c'],
#                    include_dirs=['/opt/rocm/rocprofiler/include', '/opt/rocm/roctracer/include'],
#                    library_dirs=['/opt/rocm/rocprofiler/lib', '/opt/rocm/roctracer/lib'],
#                    libraries=['roctracer64']
#)

module2 = Extension('roctxMarker',
                    sources = ['roctxMarkerModule.c'],
                    include_dirs=['/opt/rocm/include'],
                    library_dirs=['/opt/rocm/lib'],
                    libraries=['roctx64']
)

setup (name = 'HipMarker',
       version = '1.0',
       description = 'User markers for hip',
       py_modules = ['hipScopedMarker'],
       #ext_modules = [module1, module2])
       ext_modules = [module2])
