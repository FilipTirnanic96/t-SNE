# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 12:16:03 2020

@author: Filip
"""
import numpy as np
import math
import numbers
from scipy.sparse import csr_matrix
from tsne_algorithm.trees.kd_tree import KDTree
from tsne_algorithm.barnes_hut_tsne import gradient

EPSILON = np.finfo(np.double).eps


def compute_pairwise_distances(X: np.array, squared: bool):
    """
    Compute pairwise distances of data from X.

    :param X: Input data
    :param squared: Flag if squared distances
    :return: Pairwise distances
    """

    if squared:
        XX = np.sum(X ** 2, axis=1)[:, np.newaxis]
        YY = XX.T
        distance = XX - 2 * np.dot(X, X.T) + YY
    else:
        XX = np.sum(X ** 2, axis=1)[:, np.newaxis]
        YY = XX.T
        distance = np.sqrt(XX - 2 * np.dot(X, X.T) + YY)

    return distance


def check_random_state(seed: int):
    """
    Checks the random seed.

    :param seed: Seed number
    :return: Random seed
    """

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def binary_search_perplexity(distances: np.array, perplexity: int):
    """
    Binary search each gaussian sigma for each point.

    :param distances: Pairwise distances
    :param perplexity: Input perplexity
    :return: Conditional pairwise probabilities
    """

    n_samples = distances.shape[0]
    n_neighbors = distances.shape[1]
    P_conditional = np.zeros((n_samples, n_neighbors))
    goal_entropy = np.log2(perplexity)
    min_error_perplexity = 1e-5
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

            nominator = np.exp(- distances[i] * sigma_i)

            if not using_neighbors:
                nominator[i] = 0.0

            sum_Pi = np.sum(nominator)

            P_conditional[i] = nominator / np.maximum(sum_Pi, 1e-8)

            # calculate current entropy
            entropy = - np.sum(P_conditional[i] * np.log2(P_conditional[i] + 1e-10))

            entropy_diff = entropy - goal_entropy

            if np.abs(entropy_diff) < min_error_perplexity:
                break

            if entropy_diff > 0:
                sigma_min = sigma_i
                if sigma_max == math.inf:
                    sigma_i *= 2.0
                else:
                    sigma_i = (sigma_i + sigma_max) / 2.0

            else:
                sigma_max = sigma_i
                if sigma_min == -math.inf:
                    sigma_i /= 2.0
                else:
                    sigma_i = (sigma_i + sigma_min) / 2.0

        sigma_sum += sigma_i

    print("[t-SNE] Mean sigma:", np.mean(math.sqrt(n_samples / sigma_sum)))
    return P_conditional


def compute_pairwise_joint_probabilities(distances: np.array, perplexity: int):
    """
    Compute pairwise joint probabilities.

    :param distances: Pairwise distances
    :param perplexity: Input perplexity
    :return: Pairwise joint probabilities
    """

    # cast type distances array
    distances = distances.astype(np.float32, copy=False)
    # binary search pairwise affinities p_i|j with perplexity p
    P_conditional = binary_search_perplexity(distances, perplexity)
    # calculate pairwise affinities p_ij
    P = P_conditional + P_conditional.T
    # normalize p_ij
    sum_P = np.maximum(np.sum(P), EPSILON)
    P = np.maximum(P / sum_P, EPSILON)
    # p_ii = 0
    np.fill_diagonal(P, 0)

    return P


def compute_pairwise_joint_probabilities_nn(distances_csr, perplexity: int):
    """
    Compute pairwise joint probabilities.

    :param distances_csr: Pairwise distances
    :param perplexity: Input perplexity
    :return: Pairwise joint probabilities
    """

    # cast type distances array
    n_samples = distances_csr.shape[0]
    distances_csr.sort_indices()
    distances = distances_csr.data.reshape(n_samples, -1)
    distances = distances.astype(np.float32, copy=False)
    # binary search pairwise affinities p_i|j with perplexity p
    P_conditional = binary_search_perplexity(distances, perplexity)
    # calculate pairwise affinities p_ij
    P = csr_matrix((P_conditional.ravel(), distances_csr.indices, distances_csr.indptr), shape=(n_samples, n_samples))
    P = P + P.T
    # normalize p_ij
    sum_P = np.maximum(P.sum(), EPSILON)
    P = P / sum_P

    return P


def gradient_decent(y: np.array, P: np.array, it: int, n_iter: int, momentum: float = 0.5, learning_rate: float = 200.0,
                    method: str = 'exact'):
    """
    Perform gradient decent algorithm.

    :param y: Output 2d mapped data
    :param P: Pairwise joint probabilities of input data
    :param it: Current iteration
    :param n_iter: Number of iterations
    :param momentum: Gradient decent update momentum
    :param learning_rate: Learning rate of gradient decent
    :param method: Method ('exact' or 'barnes-hut')
    :return: Output data y representing input data in 2d space, current iteration, KL divergence
    """

    # init parameters
    min_gain = 0.01
    min_grad_norm = 1e-7
    update = np.zeros_like(y)
    gains = np.ones_like(y)
    kl_divergence = 0

    for i in np.arange(it, n_iter):

        # check a method for kl divergence gradient calculation
        if method == 'exact':
            kl_divergence, grad = kl_divergence_grad(y, P)
        else:
            angle = 0.5
            kl_divergence, grad = kl_divergence_bh_grad(y, P, angle)

        # use adaptive learning rate
        gains[update * grad < 0.0] += 0.2
        gains[update * grad >= 0.0] *= 0.8
        gains = np.minimum(gains, np.inf)
        gains = np.maximum(min_gain, gains)

        grad *= gains

        # update solution y(i) = y(i-1) + learning_rate * gradient(Cost func.)
        update = update * momentum - learning_rate * grad
        y = y + update
        # calculate gradient norm
        grad_norm = np.sqrt(np.sum(grad ** 2))

        if grad_norm < min_grad_norm:
            break

    return y, i, kl_divergence


def kl_divergence_bh_grad(y: np.array, P: np.array, angle: float = 0.5):
    """
    Calculate barnes hut gradient.

    :param y: Output 2d mapped data
    :param P: Pairwise joint probabilities of input data
    :param angle: Angle value for barnes-hut algorithm
    :return: KL divergence, gradient
    """

    y = y.astype(np.float32, copy=False)

    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)

    grad, kl_divergence = gradient(val_P, y, neighbors, indptr, angle)
    grad *= 4
    return kl_divergence, grad


def kl_divergence_grad(y: np.array, P: np.array):
    """
     Calculate gradient.

     :param y: Output 2d mapped data
     :param P: Pairwise joint probabilities of input data
     :return: KL divergence, gradient
     """

    # compute q(i,j)
    y_distances = compute_pairwise_distances(y, True)
    nominator = (1 + y_distances) ** (-1)
    np.fill_diagonal(nominator, 0)
    Q = nominator / (np.sum(nominator))

    # compute error
    kl_divergence = np.sum(P * np.log(np.maximum(P, EPSILON) / np.maximum(Q, EPSILON)))

    # compute gradient(Cost func.)
    PQd = (P - Q) * nominator

    grad = np.zeros_like(y)
    for j in np.arange(0, y.shape[0]):
        grad[j] = np.dot(PQd[j], y[j] - y)

    grad *= 4
    return kl_divergence, grad


def TSNE(X: np.array, perplexity: int = 30, n_iter: int = 1000, learning_rate: float = 200., early_exaggeration: float= 4.0,
         method: str = "exact", random_state: int = None, verbose: int = 0):
    """
    Calculates 2d representation of input data.

    :param X: Input data
    :param perplexity: Input perplexity
    :param n_iter: Number of iteration of gradient decent
    :param learning_rate: Value of learning rate for gradient decent
    :param early_exaggeration: Early exaggeration value
    :param method: Type of method to use ('exact' or 'barnes_hut')
    :param random_state: Random state
    :param verbose: Verbose
    :return y: 2d representation of input data
    """
    n_samples = X.shape[0]
    n_components = 2

    random_state = check_random_state(random_state)

    # compute pairwise affinities p(j|i) with defined perplexity
    if method == "exact":
        distances = compute_pairwise_distances(X, True)
        P = compute_pairwise_joint_probabilities(distances, perplexity)
    elif method == "barnes_hut":
        # Compute the number of nearest neighbors to find.
        # Use 3 * perplexity as the number of neighbors.
        n_neighbors = min(n_samples - 1, int(3. * perplexity + 1))

        # compute distances
        kd_tree = KDTree(X)

        # Take 1 more neighbour cause first is always same sample
        distances, indices = kd_tree.query(X, k=n_neighbors + 1)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        distances = distances ** 2

        # Make sparse matrix for faster calculations
        n_queries = distances.shape[0]
        num_nonzero_el = n_queries * n_neighbors
        ind_row_ptr = np.arange(0, num_nonzero_el + 1, n_neighbors)

        distances_csr = csr_matrix((distances.ravel(), indices.ravel(), ind_row_ptr), shape=(n_samples, n_samples))

        # compute pairwise joint probabilities
        P = compute_pairwise_joint_probabilities_nn(distances_csr, perplexity)
    else:
        raise ValueError("method must be 'barnes_hut' or 'exact'")

    # sample init y from Gauss ~ N(0, 10^(-4)*I)
    y = 1e-4 * random_state.randn(n_samples, n_components).astype(np.float32)

    # use gradient decent to find the solution
    # first explore for n_iter = 250, momentum = 0.5, * early_exaggeration
    n_exploration_iter = 250
    P *= early_exaggeration
    y, it, error = gradient_decent(y=y, P=P, it=0, n_iter=n_exploration_iter, momentum=0.5,
                                   learning_rate=learning_rate, method=method)
    if verbose == 1:
        print("[t-SNE] KL divergence after", it + 1, "iterations:", error)

    # use gradient decent to find the solution
    # Run remaining iteration with momentum = 0.8
    remaining_iter = n_iter - n_exploration_iter
    P /= early_exaggeration
    if remaining_iter > 0:
        y, it, error = gradient_decent(y=y, P=P, it=it + 1, n_iter=n_iter, momentum=0.8,
                                       learning_rate=learning_rate, method=method)
    if verbose == 1:
        print("[t-SNE] KL divergence after", it + 1, "iterations:", error)

    return y
