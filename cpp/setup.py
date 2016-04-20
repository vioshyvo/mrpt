# encoding: utf-8
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

from distutils.core import setup, Extension

module1 = Extension('mrpt',
                    sources = ['src/mrptmodule.cpp', 'src/mrpt.cpp', 'src/knn.cpp'],
                    extra_compile_args = ['-std=c++11'],
                    extra_link_args = ['-lblas', '-llapack'])

setup (name = 'mrpt',
       versiown = '1.0',
       description = 'This is a module package for mrpt approximate nearest neighbor search',
       ext_modules = [module1])