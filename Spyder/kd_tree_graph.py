# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 17:07:02 2020

@author: Filip
"""


from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import _kd_tree
from scipy.spatial import KDTree

def preorder(node):
    if node == None:
        return
    if isinstance(node, KDTree.innernode):
        print(node.split, node.split_dim )
        preorder(node.less)
        preorder(node.greater)

np.random.seed(50)


num_class = 5000
mean = np.array([1,1])
cov =  np.array([[1, 0.2],[0.2, 1]])
X = np.random.multivariate_normal(mean, cov, num_class)
index_arr = np.array([1, 2, 3 , 4])
arr = np.array([5, 2, 1, 6, 7 , 9, 18, 1]).reshape(-1,1)
plt.plot(X[:,0], X[:,1], 'ro')
n_samples = X.shape[0]
perplexity = 30
n_neighbors = min(n_samples - 1, int(3. * perplexity + 1))
n_neighbors = 60



kd_tree_1 = _kd_tree.KDTree(X, leaf_size=40)
t0 = time()
dist1, ind1 = kd_tree_1.query(X, k = n_neighbors + 1)
duration1 = time() - t0

data, index, bide_data, node_bounds = kd_tree_1.get_arrays()

kd_tree = KDTree(X, leafsize=40)
root = kd_tree.tree
t0 = time()
dist, ind = kd_tree.query(X, k = n_neighbors + 1)
duration2 = time() - t0
