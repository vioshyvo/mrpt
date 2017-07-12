#!/usr/bin/python

import setuptools
import numpy
from setuptools import Extension

# Not all CPUs have march as a tuning parameter
import platform
cputune, libraries, llvm = ['-march=native'], [], []
if platform.machine() == 'ppc64le':
    cputune = ['-mcpu=native']
if platform.system() == 'Darwin':
    llvm += ['-L/usr/local/opt/llvm/lib']
if platform.system() == 'Windows':
    libraries = []
else:
    libraries = ['stdc++']

setuptools.setup(
    name='mrpt',
    version='0.1',
    url='http://github.com/teemupitkanen/mrpt',
    install_requires=[],
    packages={'.': 'mrpt'},
    zip_safe=False,
    test_suite='py.test',
    entry_points='',
    ext_modules=[
        Extension('mrptlib',
            sources=[
                'cpp/mrptmodule.cpp',
            ],
            extra_compile_args=['-std=c++11', '-O3', '-ffast-math', '-s',
                                '-fno-rtti', '-fopenmp', '-DNDEBUG'] + cputune,
            libraries=libraries,
            extra_link_args=['-lgomp'] + llvm,
            include_dirs=['cpp/lib', numpy.get_include()]
        )
    ]
)
