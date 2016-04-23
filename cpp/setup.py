# encoding: utf-8
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

from distutils.core import setup, Extension
import numpy as np

module1 = Extension('mrptlib',
                    sources = ['src/mrptmodule.cpp', 'src/mrpt.cpp', 'src/knn.cpp'],
                    extra_compile_args = ['-std=c++11','-O3', '-march=native', '-ffast-math'],
                    extra_link_args = ['-lblas', '-llapack'],
                    include_dirs=[np.get_include()])

setup (name = 'mrpt',
       version = '1.0',
       description = 'This is a module for mrpt approximate nearest neighbor search',
       ext_modules = [module1])