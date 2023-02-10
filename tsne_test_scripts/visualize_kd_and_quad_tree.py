import numpy as np
from tsne_algorithm.trees import kd_tree
from tsne_algorithm.trees import quad_tree

# generate synthetic data
np.random.seed(50)
num_data = 50
mean = np.array([1, 1])
cov = np.array([[1, 0.2], [0.2, 1]])
X = np.random.multivariate_normal(mean, cov, num_data)

# visualize k-d tree on synthetic data
kd_tree = kd_tree.KDTree(X, leaf_size=10)
kd_tree.visualize_bounds()

# visualize quad tree on synthetic data
qt = quad_tree.QuadTree()
qt.build_tree(X)
qt.visualize_bounds(X)
