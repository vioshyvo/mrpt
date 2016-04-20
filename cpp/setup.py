from distutils.core import setup, Extension

module1 = Extension('mrpt',
                    sources = ['src/mrptmodule.cpp', 'src/mrpt.cpp', 'src/knn.cpp'])

setup (name = 'mrpt',
       versiown = '1.0',
       description = 'This is a module package for mrpt approximate nearest neighbor search',
       ext_modules = [module1])