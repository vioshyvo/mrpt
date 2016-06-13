# encoding: utf-8
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

from distutils.core import setup, Extension
import os
import numpy
import sys

if sys.platform == 'darwin':
    # A quick fix to make this work on my mac -Teemu
    os.environ["CC"] = "g++-6"
    os.environ["CXX"] = "g++-6"

module1 = Extension('mrptlib',
                    sources=['mrptmodule.cpp'],
                    extra_compile_args=['-std=c++11', '-Wall', '-O3', '-march=native', '-ffast-math',
                                        '-s', '-mavx', '-mfma', '-Wno-deprecated-declarations',
                                        '-Wno-ignored-attributes', '-Wno-cpp', '-fopenmp',
                                        '-fno-rtti', '-fno-stack-protector', '-fno-exceptions',
                                        '-DNDEBUG', '-DEIGEN_DONT_PARALLELIZE', '-I./lib'],
                    extra_link_args=['-lgomp', '-lpython2.7'],
                    include_dirs=[numpy.get_include()])

setup(name='mrpt',
      version='1.0',
      description='This is a module for mrpt approximate nearest neighbor search',
      ext_modules=[module1])
