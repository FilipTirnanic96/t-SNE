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
num_class_1 = 100
num_class_2 = 100

# first class Gauss 2d
mean1 = np.array([1,1])
cov1 =  np.array([[1, 0.2],[0.2, 1]])
X1 = np.random.multivariate_normal(mean1, cov1, num_class_1)
y1 = np.zeros((num_class_1))
# second class Gauss 2d
mean2 = np.array([10,1])
cov2 =  np.array([[0.1, 0],[0, 0.1]])
X2 = np.random.multivariate_normal(mean2, cov2, num_class_2)
y2 = np.ones((num_class_2))
# visualise classes
plt.plot(X1[:,0], X1[:,1], 'ro')
plt.plot(X2[:,0], X2[:,1], 'bo')
plt.xlim([-3, 13])
plt.ylim([-4, 6])
plt.legend(['class1', 'class2'])
plt.title('Sytetic data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# concat X and y
X = np.concatenate((X1, X2), axis = 0)
y = np.concatenate((y1, y2), axis = 0)

# sample class Gauss 2d for kde graph
num_class = 20
mean = np.array([1,1])
cov =  np.array([[1, 0.2],[0.2, 1]])
#X = np.random.multivariate_normal(mean, cov, num_class)
#y = np.zeros((num_class))

# visualise sample class
plt.plot(X[:,0], X[:,1], 'ro')
plt.xlim([-3, 5])
plt.ylim([-2, 5])
plt.title('Sytetic data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


n_samples = X.shape[0]
perplexity = 20
n_jobs = 1
metric = "euclidean"

n_neighbors = min(n_samples - 1, int(3. * perplexity + 1))
n_neighbors = 60
kd_tree_1 = _kd_tree.KDTree(X)
kd_tree_1.node_data
dist1, ind1 = kd_tree_1.query(X, k = n_neighbors + 1)
dist1 = dist1[:,1:]
ind1 = ind1[:,1:]


knn = NearestNeighbors(algorithm='auto',
                                   n_jobs=n_jobs,
                                   n_neighbors=n_neighbors,
                                   metric=metric)
t0 = time()
knn.fit(X)
duration = time() - t0
#kd_tree_1 = _kd_tree.KDTree(X)
kd_tree = KDTree(X, leafsize=40)
root = kd_tree.tree
dist, ind = kd_tree.query(X, k = n_neighbors + 1)
dist = dist[:,1:]
ind = ind[:,1:]
tree_arrays = kd_tree_1.get_arrays()
print("[t-SNE] Indexed {} samples in {:.3f}s...".format(
            n_samples, duration))

dist_r = dist[(dist != dist1)]
dist_r1 = dist1[(dist != dist1)]

ind_r = (ind != ind1).sum()
t0 = time()
distances_nn = knn.kneighbors_graph(mode='distance')
distance = distances_nn.data.reshape(n_samples, -1)
distance_ind = distances_nn.indices.reshape(n_samples, -1)
connectivity = knn.kneighbors_graph(mode='connectivity').toarray()
duration = time() - t0

print("[t-SNE] Computed neighbors for {} samples "
          "in {:.3f}s...".format(n_samples, duration))

# plot connectivity from dot i
i = 0
plt.plot(X[:,0], X[:,1], 'ro')
plt.plot(X[i,0], X[i,1], 'yo')
plt.plot(X[connectivity[i,:] == 1,0], X[connectivity[i,:] == 1,1], 'bo')

plt.xlim([-3, 5])
plt.ylim([-2, 5])
plt.title('Sytetic data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
   
