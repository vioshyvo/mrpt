#!/usr/bin/python

import sys

import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext

with open('requirements.txt', 'r') as fo:
    requirements = [line for line in fo]

source_files = ['cpp/mrptmodule.cpp']
ext_modules = [
    Extension(
        'mrptlib',
        source_files,
        language='c++14',
    )
]


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options.

    Assume that C++14 is available.
    """
    c_opts = {
        'msvc': ['/EHsc', '/openmp', '/O2', '/D "NDEBUG"'],
        'unix': ['-march=native', '-std=c++14', '-Ofast', '-DNDEBUG', '-fopenmp'],
    }
    link_opts = {
        'unix': ['-fopenmp', '-pthread'],
        'msvc': [],
    }
    libraries_opt = {
        'msvc': [],
        'unix': ['stdc++'],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-Xpreprocessor', '-stdlib=libc++', '-mmacosx-version-min=10.7']
        link_opts['unix'] += ['-lomp', '-stdlib=libc++', '-mmacosx-version-min=10.7']
    else:
        link_opts['unix'] += ['-lgomp']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])

        # extend include dirs here (don't assume numpy/pybind11 are installed when first run, since
        # pip could have installed them as part of executing this script
        # import pybind11
        import numpy as np
        for ext in self.extensions:
            ext.libraries.extend(self.libraries_opt.get(ct, []))
            ext.language = 'c++14'
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(self.link_opts.get(ct, []))
            ext.include_dirs.extend([
                'cpp/lib',

                # Path to numpy headers
                np.get_include()
            ])

        build_ext.build_extensions(self)


setuptools.setup(
    name='mrpt',
    author='Ville Hyvönen',
    author_email='ville.o.hyvonen@helsinki.fi',
    version='1.0',
    description='Fast nearest neighbor search with random projection',
    url='http://github.com/vioshyvo/mrpt',
    license='MIT',
    install_requires=requirements,
    setup_requires=requirements,
    packages={'.': 'mrpt'},
    zip_safe=False,
    test_suite='py.test',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
)
