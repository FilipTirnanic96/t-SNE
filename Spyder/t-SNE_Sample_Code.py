# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 11:00:20 2020

@author: uic52421
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import simple_TSNE

from sklearn.manifold import TSNE

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
# execute t-SNE
X = np.concatenate((X1, X2), axis = 0)
y = np.concatenate((y1, y2), axis = 0)
t_SNE = TSNE(n_components = 2, perplexity = 30, method="exact")
start_time = time.time()
#X_trans = t_SNE.fit_transform(X)
simple_TSNE.TSNE(X)
elapsed_time = time.time() - start_time
print("The execution of t-SNE last for ", elapsed_time, "s")
# plot t-SNE
print()
plt.plot(X_trans[np.where(y == 0),0], X_trans[np.where(y == 0),1], 'ro')
plt.plot(X_trans[np.where(y == 1),0], X_trans[np.where(y == 1),1], 'bo')
plt.legend(['class1', 'class2'])
plt.title('t-SNE on sytetic data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()