# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 12:16:03 2020

@author: uic52421
"""
import numpy as np


def compute_pairwise_distances(X, metric, squared):
    if (metric == "euclidean") :
        distance = np.zeros((X.shape[0], X.shape[0]))
        if(squared):
            for i in np.arange(0, X.shape[0]):
                 distance[i,:] = np.sum((X - X[i, :])**2, axis = 1)

        else:
            for i in np.arange(0, X.shape[0]):
                 distance[i,:] = np.sqrt(np.sum((X - X[i, :])**2, axis = 1))
                 
        return distance
        
    else:
        raise ValueError("Incorrect type of metric.")


def compute_pairwise_joint_probabilities(distances, perplexity):
    P = 0
    return P
    
    

def TSNE(X, n_components = 2, perplexity = 30, n_iter = 1000, learning_rate = 200, momentum_alpha = 0):
    # compute distances between training samples
    distances = compute_pairwise_distances(X, "euclidean")   
    
    # compute pairwise affinities p(j|i) whith perplexity Perp
    compute_pairwise_joint_probabilities(distances, perplexity)
    # compute p(i,j) =  (p(j|i) +  p(i|j)) / (2n)
    
    # sample init y from Gauss ~ N(0, 10^(-4)*I)
    y = np.random.multivariate_normal(np.zeros((X.shape[0],n_components)), 10**(-4) * np.identity(n_components), num_class_1)
    
    for i in np.arange(0, n_iter):
        # compute q(i,j)
        
        # compute gradient(Cost func.)
        
        # update solution y(i) = y(i-1) + learning_rate * gradient(Cost func.)
    
        
    return y
