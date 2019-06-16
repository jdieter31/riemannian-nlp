import re
import sysconfig
from subprocess import check_output


from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


extensions = [Extension(
        "data.graph_dataset",
        ["data/graph_dataset.pyx"],
        include_dirs=[numpy.get_include(), "."],
        extra_compile_args = ["-std=c++11", "-stdlib=libc++"],
        language='c++',
    )]


ext_modules = cythonize(extensions)
install_requires=["Cython", 'torch', 'geoopt', 'numpy']

setup(
    name='riemanniannlp',
    version='0.1',
    packages=['riemann'],
    package_dir={'riemann':'.'},
    url='',
    license='',
    author='justin',
    author_email='jdieter@stanford.edu',
    description='',
    ext_modules = ext_modules,
    include_dirs=numpy.get_include(),
    install_requires=install_requires
)
