from setuptools import setup, Extension, find_packages
import numpy
import os

from sys import platform

if platform == "win32":
    compile_args = []
else:
    compile_args = ["-std=gnu++11", "-fpic", "-g"]
if platform == "darwin":
    # To ensure gnu+11 and all std libs
    compile_args.append("-mmacosx-version-min=10.15")
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.15"

XBCF_cpp_module = Extension(
    "_xbcf_cpp_",
    sources=[
        "xbcausalforest/xbcf_wrap.cxx",
        "xbcausalforest/xbcf.cpp",
        "src/utility.cpp",
        "src/xbcf_mcmc_loop.cpp",
        "src/sample_int_crank.cpp",
        "src/common.cpp",
        "src/forest.cpp",
        "src/tree.cpp",
        "src/thread_pool.cpp",
        "src/cdf.cpp",
        "src/json_io.cpp",
        "src/xbcf_model.cpp",
    ],
    language="c++",
    include_dirs=[numpy.get_include(), ".", "src", "xbcf"],
    extra_compile_args=compile_args,
)


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="xbcausalforest",
    version="0.1.2",
    author="Jingyu He, Saar Yalov, P. Richard Hahn, Nikolay Krantsevich",
    author_email="krantsevich@gmail.com",
    description="""Implementation of Accelerated Bayesian Causal Forests""",
    long_descripition=readme(),
    include_dirs=[numpy.get_include(), ".", "src", "xbcf"],
    ext_modules=[XBCF_cpp_module],
    install_requires=["numpy"],
    license="Apache-2.0",
    py_modules=["xbcf"],
    python_requires=">3.7",
    packages=find_packages(),
)
