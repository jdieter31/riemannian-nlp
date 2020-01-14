import os
import sys

import sysconfig
from subprocess import check_output
from Cython.Build import cythonize
from setuptools import Extension
import numpy as np

def is_clang():
    """
    Super hacky way of determining if clang or gcc is being used
    """
    CC = sysconfig.get_config_vars().get('CC', 'gcc').split(' ')[0]
    out = check_output([CC, '--version'])
    return re.search('apple *llvm', str(out.lower()))

# 1. 
extra_compile_args = ['-std=c++11']
if is_clang():
    extra_compile_args.append('-stdlib=libc++')

# use cythonize to build the extensions
extensions = [Extension(
        "riemann.data.graph_dataset",
        ["riemann/data/graph_dataset.pyx"],
        include_dirs=[numpy.get_include(), "."],
        extra_compile_args = extra_compile_args,
        language='c++',
    )]
ext_modules = cythonize(extensions)

def build(setup_kwargs):
    """Needed for the poetry building interface."""

    setup_kwargs.update({
        'ext_modules' : ext_modules,
        'include_dirs' : [np.get_include()],
    })
