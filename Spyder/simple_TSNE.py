# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 12:16:03 2020

@author: uic52421
"""
import numpy as np
import math
from time import time
import numbers
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
    """  dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y)) - much faster"""
    n_samples = X.shape[0]
    if (metric == "euclidean") :
        distance = np.zeros((n_samples, n_samples))
        if(squared):
            XX = np.sum(X**2, axis = 1)[:,np.newaxis]
            YY = XX.T
            distance = XX - 2*np.dot(X,X.T) + YY
        else:
            XX = np.sum(X**2, axis = 1)[:,np.newaxis]
            YY = XX.T
            distance = np.sqrt(XX - 2*np.dot(X,X.T) + YY)
        '''if(squared):
            for i in np.arange(0, n_samples):
                # distance[i,:] = np.sum((X - X[i, :])**2, axis = 1)
                distance[i,:] = np.sum((X - X[i, :])**2, axis = 1)
        else:
            for i in np.arange(0, n_samples):
                 distance[i,:] = np.sqrt(np.sum((X - X[i, :])**2, axis = 1))
         '''        
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
    P = np.maximum(P / sum_P, 0)
    np.fill_diagonal(P, 0)
    return P
   
    
def gredient_decent(y, P, it, n_iter, n_iter_without_progress=300, n_iter_check = 50, momentum=0.5, learning_rate=200.0, min_gain=0.01, min_grad_norm=1e-7):
    grad = np.zeros_like(y)
    update = np.zeros_like(y)
    gains = np.ones_like(y)
    for i in np.arange(it, n_iter):
        # compute q(i,j)
        y_distances = compute_pairwise_distances(y, "euclidean", True) 
        nominator = (1 + y_distances) ** (-1)
        np.fill_diagonal(nominator, 0)
        Q = nominator / (np.sum(nominator))
        
        # compute error
        if (i + 1) % n_iter_check == 0 or i == n_iter - 1:
            kl_divergence =  np.sum(P * np.log(np.maximum(P, 1e-16) / np.maximum(Q, 1e-16)))
        else:
            kl_divergence = np.nan

        # compute gradient(Cost func.)
        PQd = (P - Q) * nominator
        
        
        for j in np.arange(0, y.shape[0]):
            grad[j] =  np.dot(PQd[j], y[j] - y)      
        grad *= 4 
        
        gains[update * grad < 0.0] += 0.2
        gains[update * grad > 0.0] *= 0.8
        gains = np.minimum(gains, np.inf)
        gains = np.maximum(min_gain, gains)
        
        grad *= gains
        # update solution y(i) = y(i-1) + learning_rate * gradient(Cost func.)
        update = update * momentum - learning_rate * grad 
        y = y + update 
        grad_norm = np.sqrt(np.sum(grad**2))
        if grad_norm < min_grad_norm:
            break;
    return y, i, kl_divergence


def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def TSNE(X, n_components = 2, perplexity = 30, n_iter = 1000, learning_rate = 200, early_exaggeration=4.0, method = "exact", random_state = None, verbose = 0):
    n_samples = X.shape[0]
    # compute distances between training samples
    start_time = time()  
    elapsed_time = time() - start_time
    random_state = check_random_state(random_state)
    #print("The execution of compute_pairwise_distances() simple t-SNE last for ", elapsed_time, "s") 
    # compute pairwise affinities p(j|i) whith perplexity Perp
    #start_time = time()
    if method == "exact":
        distances = compute_pairwise_distances(X, "euclidean", True) 
    else:
        # make NearestNeighbors graph
        
        # Compute the number of nearest neighbors to find.
        # LvdM uses 3 * perplexity as the number of neighbors.
        # In the event that we have very small # of points
        # set the neighbors to n - 1.
        n_neighbors = min(n_samples - 1, int(3. * perplexity + 1))
        
        # compute distances
    
    P = compute_pairwise_joint_probabilities(distances, perplexity)
    #elapsed_time = time() - start_time
    #print("The execution of compute_pairwise_joint_probabilities() simple t-SNE last for ", elapsed_time, "s") 
    # sample init y from Gauss ~ N(0, 10^(-4)*I)
    y = 1e-4 * random_state.randn(n_samples, n_components).astype(np.float32)
    # use gradinet decent to find the solution    
    # first explore for n_iter = 250, momentum = 0.5, early_exaggeration = 4
    n_exploration_iter = 250
    P *= early_exaggeration
    y, it, error = gredient_decent(y = y, P = P, it = 0, n_iter = n_exploration_iter, momentum = 0.5)
    if verbose == 1:
        print("[t-SNE] KL divergence after %d iterations: %f"% (it + 1, error))
    # Run remaining iteration with momentum = 0.5
    remaining_iter = n_iter - n_exploration_iter
    P /= early_exaggeration
    if remaining_iter > 0:
        y, it, error = gredient_decent(y = y, P = P, it = it + 1, n_iter = n_iter, momentum = 0.8)
    if verbose == 1:
        print("[t-SNE] KL divergence after %d iterations: %f"% (it + 1, error))
   
    return y
