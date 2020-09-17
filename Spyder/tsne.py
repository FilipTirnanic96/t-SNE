# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 12:16:03 2020

@author: Filip
"""
import numpy as np
import math
import numbers
from scipy.sparse import csr_matrix
from trees.kd_tree import KDTree
from barnes_hut_tsne import gradient


EPSILON = np.finfo(np.double).eps
  
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
            
        #np.fill_diagonal(distance, 0)
        return distance

    else:
        raise ValueError("Incorrect type of metric.")

def binary_search_perplexity(distances, perplexity):

    n_samples = distances.shape[0]
    n_neighbors = distances.shape[1]
    P_conditional = np.zeros((n_samples, n_neighbors))
    goal_entropy = np.log2(perplexity)
    min_error_preplexity = 1e-5
    n_iter = 100
    
    using_neighbors = n_neighbors < n_samples
    sigma_sum = 0.0
    # for each sample calculate optimal sigma_i
    for i in np.arange(0, n_samples):
        sigma_max = np.inf
        sigma_min = -np.inf
        sigma_i = 1.0

        # binary search for sigma_i
        for j in np.arange(0, n_iter):
            
            try:
                nominator = np.exp(- distances[i] * sigma_i)
            except:
                print("Error")
                
            if not using_neighbors:
                nominator[i] = 0.0
                
            sum_Pi = np.sum(nominator)
            if  sum_Pi == 0:
                sum_Pi = 1e-8
                
            P_conditional[i] = nominator / sum_Pi

            #calculate current entropy
            entropy =  - np.sum(P_conditional[i] * np.log2(P_conditional[i] + 1e-10))

            entropy_diff = entropy - goal_entropy
            
            if np.abs(entropy_diff) < min_error_preplexity:
                break
            
            if entropy_diff > 0:
                sigma_min = sigma_i
                if sigma_max == math.inf:
                    sigma_i *=  2.0
                else:
                    sigma_i = (sigma_i + sigma_max)/ 2.0
                
            else:
                sigma_max = sigma_i
                if sigma_min == -math.inf:
                    sigma_i /= 2.0
                else:
                    sigma_i = (sigma_i + sigma_min)/ 2.0
    
        sigma_sum += sigma_i
    
    print("[t-SNE] Mean sigma:" ,np.mean(math.sqrt(n_samples / sigma_sum)))
    return P_conditional
  
  
def compute_pairwise_joint_probabilities(distances, perplexity):
    distances = distances.astype(np.float32, copy=False)
    P_conditional = binary_search_perplexity(distances, perplexity)
    P = P_conditional + P_conditional.T
    sum_P = np.maximum(np.sum(P), EPSILON)
    P = np.maximum(P / sum_P, EPSILON)
    np.fill_diagonal(P, 0)
    return P
   
def compute_pairwise_joint_probabilities_nn(distances_csr, perplexity):
    n_samples = distances_csr.shape[0]
    
    distances_csr.sort_indices()
    distances = distances_csr.data.reshape(n_samples, -1)
    distances = distances.astype(np.float32, copy=False)
    P_conditional = binary_search_perplexity(distances, perplexity) 
    
    P = csr_matrix((P_conditional.ravel(), distances_csr.indices, distances_csr.indptr), shape=(n_samples, n_samples))
    P = P + P.T
    
    sum_P = np.maximum(P.sum(), EPSILON)
    P = P / sum_P
    
    return P

  
def gredient_decent(y, P, it, n_iter, n_iter_without_progress=300, n_iter_check = 50, momentum=0.5, learning_rate=200.0, min_gain=0.01, min_grad_norm=1e-7, method = 'exact', angle=0.5):
    update = np.zeros_like(y)
    gains = np.ones_like(y)
    for i in np.arange(it, n_iter):
        
        if(method == 'exact'):
            kl_divergence, grad = kl_divergence_grad(y, P, n_iter, n_iter_check, i)
        else:
            kl_divergence, grad = kl_divergence_bh_grad(y, P, y.shape[0], y.shape[1], angle)
        # adaptive learning rate  
        gains[update * grad < 0.0] += 0.2
        gains[update * grad >= 0.0] *= 0.8
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


def kl_divergence_bh_grad(y, P, n_samples, n_components,
                      angle=0.5, verbose=False,
                      compute_error=True, num_threads=1):
        y = y.astype(np.float32, copy=False)
    
        val_P = P.data.astype(np.float32, copy=False)
        neighbors = P.indices.astype(np.int64, copy=False)
        indptr = P.indptr.astype(np.int64, copy=False)
    
        grad, error = gradient(val_P, y, neighbors, indptr,
                                          angle, n_components, verbose,
                                          compute_error=compute_error)        
        grad *= 4  
        return error, grad
    
    
    

    
def kl_divergence_grad(y, P, n_iter, n_iter_check, i):
        
        # compute q(i,j)
        y_distances = compute_pairwise_distances(y, "euclidean", True) 
        nominator = (1 + y_distances) ** (-1)
        np.fill_diagonal(nominator, 0)
        Q = nominator / (np.sum(nominator))
        
        # compute error
        if (i + 1) % n_iter_check == 0 or i == n_iter - 1:
            kl_divergence =  np.sum(P * np.log(np.maximum(P, EPSILON) / np.maximum(Q, EPSILON)))
        else:
            kl_divergence = np.nan

        # compute gradient(Cost func.)
        PQd = (P - Q) * nominator
        
        grad = np.zeros_like(y)
        for j in np.arange(0, y.shape[0]):
            grad[j] =  np.dot(PQd[j], y[j] - y)   
            
        grad *= 4     
        return kl_divergence, grad
    
def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def TSNE(X, n_components = 2, perplexity = 30, n_iter = 1000, learning_rate = 200, early_exaggeration=4.0, method = "exact", random_state = None, verbose = 0, angle=0.5):
    n_samples = X.shape[0]
    metric = "euclidean"
    
    # compute distances between training samples
    random_state = check_random_state(random_state)
    
    # compute pairwise affinities p(j|i) whith perplexity Perp
    if method == "exact":
        distances = compute_pairwise_distances(X, "euclidean", True)
        P = compute_pairwise_joint_probabilities(distances, perplexity)
    elif method == "barnes_hut":
        if(n_components!=2):
            raise ValueError("If selected method is 'barnes_hut' number of dimensions should be 2.")
        # make NearestNeighbors graph
        
        # Compute the number of nearest neighbors to find.
        # LvdM uses 3 * perplexity as the number of neighbors.
        # In the event that we have very small # of points
        # set the neighbors to n - 1.
        n_neighbors = min(n_samples - 1, int(3. * perplexity + 1))

        # compute distances
        kd_tree = KDTree(X)
        
        # Take 1 more neigbour cause first is always same sample
        distances, indices = kd_tree.query(X, k = n_neighbors + 1)
        distances = distances[:,1:]
        indices = indices[:,1:]
        
        if metric == "euclidean":
            distances = distances**2
        
        
        # Make sparse matrix for faster calculations
        n_queries = distances.shape[0]
        num_nonzero_el = n_queries * n_neighbors
        ind_row_ptr = np.arange(0, num_nonzero_el + 1, n_neighbors)    
        
        distances_csr = csr_matrix((distances.ravel(), indices.ravel(), ind_row_ptr), shape=(n_samples, n_samples))
        
        # compute distances
         
        P = compute_pairwise_joint_probabilities_nn(distances_csr, perplexity)
    else:
         raise ValueError("method must be 'barnes_hut' or 'exact'")
    

    # sample init y from Gauss ~ N(0, 10^(-4)*I)
    y = 1e-4 * random_state.randn(n_samples, n_components).astype(np.float32)
    # use gradinet decent to find the solution    
    # first explore for n_iter = 250, momentum = 0.5, early_exaggeration = 4
    n_exploration_iter = 250
    P *= early_exaggeration
    y, it, error = gredient_decent(y = y, P = P, it = 0, n_iter = n_exploration_iter, momentum = 0.5, method = method, angle = angle)
    if verbose == 1:
        print("[t-SNE] KL divergence after",it + 1, "iterations:", error)
    # Run remaining iteration with momentum = 0.8
    remaining_iter = n_iter - n_exploration_iter
    P /= early_exaggeration
    if remaining_iter > 0:
        y, it, error = gredient_decent(y = y, P = P, it = it + 1, n_iter = n_iter, momentum = 0.8, method = method, angle = angle)
    if verbose == 1:
        print("[t-SNE] KL divergence after",it + 1, "iterations:", error)
   
    return y
