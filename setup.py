#!/usr/bin/python

import setuptools
import numpy
from setuptools import Extension

setuptools.setup(
    name='mrpt',
    version='0.1',
    url='http://github.com/teemupitkanen/mrpt',
    install_requires=[],
    packages={ '.': 'mrpt' },
    zip_safe=False,
    test_suite='py.test',
    entry_points='',
    ext_modules = [
        Extension('mrptlib',
            sources = [
                'cpp/mrptmodule.cpp',
            ],
            extra_compile_args=['-std=c++11', '-O3', '-march=native', '-ffast-math', '-s',
                                '-Wno-deprecated-declarations', '-Wno-ignored-attributes', '-Wno-cpp',
                                '-Wno-unused-result', '-fno-rtti', '-fopenmp', '-DNDEBUG'],
            extra_link_args=['-lgomp'],
            libraries = ['stdc++'],
            include_dirs = ['cpp/lib', numpy.get_include()]
        )
    ]
)
