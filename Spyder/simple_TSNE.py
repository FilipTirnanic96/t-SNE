# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 12:16:03 2020

@author: uic52421
"""
import numpy as np
import math

"""
Compute distances between all dataset points

Parameters
----------
X : array, shape (n_samples* n_features)
metric: string
squared: boolean

Returns
-------
distances : array, shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
"""
def compute_pairwise_distances(X, metric, squared):
    n_samples = X.shape[0]
    if (metric == "euclidean") :
        distance = np.zeros((n_samples, n_samples))
        if(squared):
            for i in np.arange(0, n_samples):
                 distance[i,:] = np.sum((X - X[i, :])**2, axis = 1)

        else:
            for i in np.arange(0, n_samples):
                 distance[i,:] = np.sqrt(np.sum((X - X[i, :])**2, axis = 1))
                 
        return distance

    else:
        raise ValueError("Incorrect type of metric.")


def compute_pairwise_joint_probabilities(distances, perplexity, n_iter = 100, min_error_preplexity = 1e-4):
   
    n_samples = distances.shape[0]
    P_conditional = np.zeros((n_samples, n_samples))
    goal_entropy = np.log2(perplexity)

    # for each sample calculate optimal sigma_i
    for i in np.arange(0, n_samples):
        sigma_max = math.inf
        sigma_min = -math.inf
        sigma_i = 1
        
        # binary search for sigma_i
        for j in np.arange(0, n_iter):
            try:
                nominator = np.exp(- distances[i,:] / (2*sigma_i**2))
            except:
                print("Error")
            nominator[i] = 1e-10
            sum_Pi = np.sum(nominator)
            if  sum_Pi == 0:
                sum_Pi = 1e-10
                
            P_conditional[i,:] = nominator / sum_Pi
            
            # Check if this is running faster
            #sum_distance =  np.sum(P_conditional[i,:] * distances[i,:])
            # entropy = np.log2(sum_Pi) + sum_distance  / (2*sigma_i**2)
            
            entropy =  - np.sum(P_conditional[i,:] * np.log2(P_conditional[i,:] + 1e-10))
            entropy_diff = entropy - goal_entropy
            
            if np.abs(entropy_diff) < min_error_preplexity:
                break
            
            if entropy_diff < 0:
                sigma_min = sigma_i
                if sigma_max == math.inf:
                    sigma_i = sigma_i * 2
                else:
                    sigma_i = (sigma_i + sigma_max)/ 2
                
            else:
                sigma_max = sigma_i
                if sigma_min == -math.inf:
                    sigma_i = sigma_i / 2
                else:
                    sigma_i = (sigma_i + sigma_min)/ 2
            
    P = P_conditional + P_conditional.T
    sum_P = np.maximum(np.sum(P), 1e-7)
    P = np.maximum(P / sum_P, 1e-7)
    return P
   
    
def gredient_decent(y, P, n_iter, n_iter_without_progress=300, momentum=0.8, learning_rate=200.0, min_gain=0.01, min_grad_norm=1e-7):
    grad = np.zeors((y.shape[0], y.shape[1]))
    for i in np.arange(0, n_iter):
        # compute q(i,j)
        y_distances = compute_pairwise_distances(y, "euclidean", True) 
        nominator = (1 + y_distances) ** (-1)
        np.fill_diagonal(nominator, 0)
        Q = nominator / (np.sum(nominator))
        # compute gradient(Cost func.)
        PQd = (P - Q) * nominator
        for j in range(0, y.shape[0]):
            grad[j] = np.dot(np.ravel(PQd[i], order='K'), y[i] - y)
        # update solution y(i) = y(i-1) + learning_rate * gradient(Cost func.)
        
    return y

def TSNE(X, n_components = 2, perplexity = 30, n_iter = 1000, learning_rate = 200, momentum = 0):
    # compute distances between training samples
    distances = compute_pairwise_distances(X, "euclidean", True)   
    
    # compute pairwise affinities p(j|i) whith perplexity Perp
    P = compute_pairwise_joint_probabilities(distances, perplexity)
     
    # sample init y from Gauss ~ N(0, 10^(-4)*I)
    y = np.random.multivariate_normal(np.zeros((2)), 10**(-4) * np.identity(2), X.shape[0])
    
    # use gradinet decent to find the solution    
    y = gredient_decent(p0 = y, p = P, n_iter = n_iter)
        
    return y
