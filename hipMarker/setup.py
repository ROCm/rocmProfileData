from distutils.core import setup, Extension

module1 = Extension('hipMarker',
                    sources = ['hipMarkerModule.c'],
                    include_dirs=['/opt/rocm/rocprofiler/include', '/opt/rocm/roctracer/include'],
                    library_dirs=['/opt/rocm/rocprofiler/lib', '/opt/rocm/roctracer/lib'],
                    libraries=['roctracer64']
)

setup (name = 'HipMarker',
       version = '1.0',
       description = 'User markers for hip',
       py_modules = ['hipScopedMarker'],
       ext_modules = [module1])
