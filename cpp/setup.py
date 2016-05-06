# encoding: utf-8
#
# Author: Teemu Henrikki Pitkänen <teemu.pitkanen@helsinki.fi>
# University of Helsinki / Helsinki Institute for Information Technology 2016
#

from distutils.core import setup, Extension

module1 = Extension('mrptlib',
                    sources = ['mrptmodule.cpp', 'Mrpt.cpp'],
                    extra_compile_args = ['-std=c++11','-O3', '-march=native', '-ffast-math'],
                    extra_link_args = ['-lblas', '-llapack'])

setup (name = 'mrpt',
       version = '1.0',
       description = 'This is a module for mrpt approximate nearest neighbor search',
       ext_modules = [module1])