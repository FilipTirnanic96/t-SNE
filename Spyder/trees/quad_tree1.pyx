# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 20:04:46 2020

@author: Filip
"""

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
# Author: Thomas Moreau <thomas.moreau.2010@gmail.com>
# Author: Olivier Grisel <olivier.grisel@ensta.fr>


from sklearn.neighbors._quad_tree cimport _QuadTree
cimport numpy as np
import numpy as np
ctypedef np.float32_t DTYPE_t

def _py_summarize(_QuadTree qt, DTYPE_t[:] query_pt, DTYPE_t[:, :] X, float angle):
    # Used for testing summarize
    cdef:
        DTYPE_t[:] summary
        int n_samples, n_dimensions

    n_samples = X.shape[0]
    n_dimensions = X.shape[1]
    summary = np.empty(4 * n_samples, dtype=np.float32)

    idx = qt.summarize(&query_pt[0], &summary[0], angle * angle)
    return idx, np.asarray(summary)