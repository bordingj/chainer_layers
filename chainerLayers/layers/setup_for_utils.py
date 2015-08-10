# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 11:50:47 2015

@author: bordingj
"""

import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

ext_modules = cythonize(
           "utils.pyx",                 # our Cython source
           #language="c++",             # generate C++ code
           #sources=["*.cpp"],  # additional source file(s)
           include_path = [np.get_include()]
      )

extra_compile_args = ['-fopenmp']
extra_link_args = ['-fopenmp']
for e in ext_modules:
    e.extra_compile_args.extend(extra_compile_args)
    e.extra_link_args.extend(extra_link_args)
    
setup(ext_modules = ext_modules)