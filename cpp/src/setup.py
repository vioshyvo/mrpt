from distutils.core import setup, Extension

module1 = Extension('mrpt',
                    sources = ['mrptmodule.cpp', 'mrpt.cpp', 'knn.cpp'])

setup (name = 'mrpt',
       version = '1.0',
       description = 'This is a module package for mrpt approximate nearest neighbor search',
       ext_modules = [module1])