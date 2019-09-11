from distutils.core import setup, Extension

module1 = Extension('hipMarker',
                    sources = ['hipMarkerModule.c'])

setup (name = 'HipMarker',
       version = '1.0',
       description = 'User markers for hip',
       ext_modules = [module1])
