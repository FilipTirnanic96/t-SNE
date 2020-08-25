# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 20:53:39 2020

@author: Filip
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("kd_tree1.pyx"),
    include_dirs=[numpy.get_include()]
)