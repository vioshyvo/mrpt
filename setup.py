#!/usr/bin/python

import sys
import codecs

import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext

with codecs.open("README.md", encoding="utf-8") as f:
    long_description = f.read()

ext_modules = [
    Extension(
        "mrptlib",
        ["cpp/mrptmodule.cpp"],
        language="c++",
    )
]


def has_flag(compiler, flagname):
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False

    return True


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options.

    Assume that C++14 is available.
    """

    c_opts = {
        "unix": [
            "-std=c++14",
            "-O3",
            "-fPIC",
            "-DNDEBUG",
            "-DEIGEN_DONT_PARALLELIZE",
            "-Wl,--no-undefined",
        ],
        "msvc": [
            "/std:c++14",
            "/O2",
            "/EHsc",
            "/DNDEBUG",
            "/DEIGEN_DONT_PARALLELIZE",
            "/wd4244",
        ],
    }
    link_opts = {
        "unix": ["-pthread"],
        "msvc": [],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.link_opts.get(ct, [])

        if ct == "unix":
            opts.extend(
                [
                    "-fassociative-math",
                    "-fno-signaling-nans",
                    "-fno-trapping-math",
                    "-fno-signed-zeros",
                    "-freciprocal-math",
                    "-fno-math-errno",
                ]
            )

            for flag in ["-fvisibility=hidden", "-march=native", "-mcpu=native"]:
                if has_flag(self.compiler, flag):
                    opts.append(flag)

            if sys.platform == "darwin":
                opts.extend(["-stdlib=libc++", "-mmacosx-version-min=11.0"])
                link_opts.extend(["-stdlib=libc++", "-mmacosx-version-min=11.0"])

                if has_flag(self.compiler, "-fopenmp"):
                    opts.append("-fopenmp")
                    link_opts.append("-lomp")
            else:
                opts.append("-fopenmp")
                link_opts.append("-lgomp")
        elif ct == "msvc":
            opts.append("/openmp")

        import numpy as np

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(link_opts)
            ext.include_dirs.extend(
                [
                    "cpp/lib",
                    # Path to numpy headers
                    np.get_include(),
                ]
            )

        build_ext.build_extensions(self)


setuptools.setup(
    name="mrpt",
    author="Ville Hyvönen",
    author_email="ville.o.hyvonen@helsinki.fi",
    version="2.0.1",
    description="Fast nearest neighbor search with random projection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/vioshyvo/mrpt",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Database :: Database Engines/Servers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords="vector search, approximate nearest neighbor search",
    packages={".": "mrpt"},
    zip_safe=False,
    ext_modules=ext_modules,
    install_requires=["numpy"],
    test_suite="py.test",
    cmdclass={"build_ext": BuildExt},
)
