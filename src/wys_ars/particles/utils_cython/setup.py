from setuptools import setup
from Cython.Build import cythonize
#import Cython.Compiler.Options
#
#Cython.Compiler.Options.annotate = True

setup(
    name="pairwise velocity routines",
    ext_modules=cythonize("pairwise_velocity.pyx"),
    #package_dir={'cython_test': './utils_cython/'},
    #package_data={'test': ["./utils_cython/"]},
    zip_safe=False,
)
