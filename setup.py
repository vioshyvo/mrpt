#!/usr/bin/python

import setuptools
import numpy
from setuptools import Extension

# Not all CPUs have march as a tuning parameter
import platform
cputune, libraries, fopenmp, llvm = ['-march=native'], [], ['-fopenmp'], ['-lgomp']
if platform.machine() == 'ppc64le':
    cputune = ['-mcpu=native']
if platform.system() == 'Darwin':
    fopenmp = ['-Xpreprocessor', '-fopenmp']
    llvm = ['-lomp']
if platform.system() == 'Windows':
    libraries = []
else:
    libraries = ['stdc++']

setuptools.setup(
    name='mrpt',
    version='1.0',
    url='http://github.com/vioshyvo/mrpt',
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
            extra_compile_args=['-std=c++11', '-Ofast', '-DNDEBUG'] + cputune + fopenmp,
            libraries=libraries,
            extra_link_args=llvm,
            include_dirs=['cpp/lib', numpy.get_include()]
        )
    ]
)
