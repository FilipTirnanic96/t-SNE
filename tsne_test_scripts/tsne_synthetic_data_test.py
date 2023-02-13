# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 11:00:20 2020

@author: Filip
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time
import tsne_algorithm.tsne
from sklearn.manifold import TSNE

np.random.seed(50)


def generate_2d_synthetic_data(n_class: int, mean: np.array, cov: np.array, class_label: int):

    X = np.random.multivariate_normal(mean, cov, n_class)
    y = class_label * np.ones(n_class)

    return X, y


def visualise_tSNE_data(X: np.array, y: np.array, title: str):
    # plot t-SNE
    plt.plot(X[np.where(y == 0), 0], X[np.where(y == 0), 1], 'ro')
    plt.plot(X[np.where(y == 1), 0], X[np.where(y == 1), 1], 'bo')
    plt.legend(['class1', 'class2'])
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


# number of samples in each class
num_class_1 = 100
num_class_2 = 100

# first class Gauss 2d
mean1 = np.array([1, 1])
cov1 = np.array([[1, 0.2], [0.2, 1]])
X1, y1 = generate_2d_synthetic_data(num_class_1, mean1, cov1, 0)

# second class Gauss 2d
mean2 = np.array([10, 1])
cov2 = np.array([[0.1, 0], [0, 0.1]])
X2, y2 = generate_2d_synthetic_data(num_class_2, mean2, cov2, 1)

# visualise classes
plt.plot(X1[:, 0], X1[:, 1], 'ro')
plt.plot(X2[:, 0], X2[:, 1], 'bo')
plt.xlim([-3, 13])
plt.ylim([-4, 6])
plt.legend(['class1', 'class2'])
plt.title('Synthetic data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# concatenate data
X = np.concatenate((X1, X2), axis = 0)
y = np.concatenate((y1, y2), axis = 0)

# execute t-SNE
t_SNE = TSNE(n_components = 2, n_iter = 1000, method = "barnes_hut", perplexity = 30, early_exaggeration = 1, verbose = 1, random_state = 1)
start_time = time()
X_trans_t = t_SNE.fit_transform(X)
elapsed_time = time() - start_time
print("The execution of t-SNE last for ", elapsed_time, "s")

# plot t-SNE
visualise_tSNE_data(X_trans_t, y, 't-SNE on synthetic data')

start_time = time()
X_trans = tsne_algorithm.tsne.TSNE(X, perplexity = 30, n_iter = 1000, early_exaggeration = 1, method = "barnes_hut", random_state = 1, verbose = 1)
elapsed_time = time() - start_time
print("The execution of implemented t-SNE last for ", elapsed_time, "s")

# plot t-SNE
visualise_tSNE_data(X_trans, y, 'Implemented t-SNE on synthetic data')

