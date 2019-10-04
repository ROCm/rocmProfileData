from distutils.core import setup, Extension

module1 = Extension('hipMarker',
                    sources = ['hipMarkerModule.c'],
                    include_dirs=['/data/Downloads/rocprofiler_pkg/inc'],
                    library_dirs=['/data/Downloads/rocprofiler_pkg/lib'],
                    libraries=['roctracer64']
)

setup (name = 'HipMarker',
       version = '1.0',
       description = 'User markers for hip',
       py_modules = ['hipScopedMarker'],
       ext_modules = [module1])
