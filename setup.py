#!/usr/bin/python

import os
import sys
import codecs
import platform
import subprocess
from pathlib import Path

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


def clang_info(compiler):
    try:
        cmd = getattr(compiler, "compiler", None) or getattr(compiler, "compiler_so", None) or []
        if not cmd:
            return False
        out = subprocess.check_output(cmd + ["--version"], stderr=subprocess.STDOUT)
        txt = out.decode("utf-8", errors="ignore").lower()
        is_clang = "clang" in txt and "gcc" not in txt
        is_apple_clang = "apple clang" in txt and "gcc" not in txt
        return is_clang, is_apple_clang
    except Exception:
        return False, False


def get_architecture():
    machine = platform.machine().lower()

    if machine in ["x86_64", "amd64", "i386", "i686"]:
        return "x86-64" if "64" in machine else "x86"
    elif machine in ["arm64", "aarch64"]:
        return "arm64"
    elif "arm" in machine:
        return "arm"
    else:
        return machine


def native_flags(compiler):
    c = compiler.lower()
    a = get_architecture()

    mn = ["-march=native", "-mtune=native"]
    mc = ["-mcpu=native"]

    flags = {
        "gcc": {"x86-64": mn, "arm64": mc},
        "clang": {"x86-64": mn, "arm64": mc},
    }
    defaults = {
        "gcc": mn,
        "clang": mc,
    }

    if c not in flags:
        raise ValueError(f"Unknown compiler '{compiler}'")

    return flags[c].get(a, defaults[c])


def macos_openmp_flags(is_clang, is_apple_clang):
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and Path(conda_prefix, "lib", "libomp.dylib").exists():
        prefix = Path(conda_prefix)
    else:
        prefix = Path("/opt/homebrew/opt/libomp")

    include_dir = prefix / "include"
    lib_dir = prefix / "lib"

    if is_clang:
        if is_apple_clang:
            opts = ["-Xpreprocessor", "-fopenmp", f"-I{include_dir}"]
            link_opts = [f"-L{lib_dir}", "-lomp", f"-Wl,-rpath,{lib_dir}"]
            return opts, link_opts
        else:
            opts = ["-fopenmp", f"-I{include_dir}"]
            link_opts = [f"-L{lib_dir}", "-fopenmp", "-lomp", f"-Wl,-rpath,{lib_dir}"]
            return opts, link_opts

    opts = ["-fopenmp", f"-I{include_dir}"]
    link_opts = [f"-L{lib_dir}", "-fopenmp", f"-Wl,-rpath,{lib_dir}"]
    return opts, link_opts



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

            is_clang, is_apple_clang = clang_info(self.compiler)
            native = native_flags("clang" if is_clang else "gcc")

            for flag in native + ["-fvisibility=hidden"]:
                if has_flag(self.compiler, flag):
                    opts.append(flag)

            if sys.platform == "darwin":
                opts.append("-mmacosx-version-min=11.0")
                link_opts.append("-mmacosx-version-min=11.0")

                if is_clang and has_flag(self.compiler, "-stdlib=libc++"):
                    opts.append("-stdlib=libc++")
                    link_opts.append("-stdlib=libc++")

                clang_opts, clang_link_opts = macos_openmp_flags(is_clang, is_apple_clang)
                opts.extend(clang_opts)
                link_opts.extend(clang_link_opts)
            else:
                opts.append("-fopenmp")
                link_opts.append("-fopenmp")
                if is_clang:
                    link_opts.append("-lomp")

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
    version="2.0.2",
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
