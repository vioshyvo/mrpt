# encoding: utf-8
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

from distutils.core import setup, Extension

module1 = Extension('mrptlib',
                    sources = ['mrptmodule.cpp'],
                    extra_compile_args = ['-std=c++11', '-Wall', '-O3', '-march=native', '-ffast-math',
                                          '-s', '-mavx', '-mfma', '-Wno-deprecated-declarations',
                                          '-Wno-ignored-attributes', '-Wno-cpp', '-fopenmp',
                                          '-fno-rtti', '-fno-stack-protector', '-fno-exceptions',
                                          '-DNDEBUG', '-DEIGEN_DONT_PARALLELIZE', '-I./lib'],
                    extra_link_args = ['-Wl,-z,defs', '-lgomp'])

setup (name = 'mrpt',
       version = '1.0',
       description = 'This is a module for mrpt approximate nearest neighbor search',
       ext_modules = [module1])
