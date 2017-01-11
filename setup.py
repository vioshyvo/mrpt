#!/usr/bin/python

import setuptools
import numpy
from setuptools import Extension

setuptools.setup(
    name='mrpt',
    version='0.01',
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
            libraries = ['stdc++'],
            include_dirs = ['cpp/lib', numpy.get_include()]
        )
    ]

)

