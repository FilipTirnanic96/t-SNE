# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 18:05:48 2020

@author: Filip
"""
import numpy as np
from tsne_algorithm.trees.quad_tree import QuadTree
from time import time
FLOAT64_EPS = np.finfo(np.float64).eps
FLOAT32_TINY = np.finfo(np.float32).tiny


def gradient(P: np.array, y: np.array, neighbors_idx: np.array, indptr_P: np.array, theta: float):
    """
    Computes gradient with barnes-hut algorithm.

    :param P: Pairwise joint probability of high dimension input data
    :param y: Input data
    :param neighbors_idx: Neighbourhood y point indices
    :param indptr_P: Index of non 0 values
    :param theta: Threshold for summarizing data
    :return: calculated negative forces
    """

    qt = QuadTree()
    qt.build_tree(y)

    #start_time = time()
    negative_forces, sum_Q = negative_forces_gradient(y, qt, theta)
    #print("Neg forces", time() - start_time)
    positive_forces, error = positive_forces_gradient(P, y, neighbors_idx, indptr_P, sum_Q)
    total_forse = positive_forces - negative_forces/sum_Q
    return total_forse, error


def negative_forces_gradient(y: np.array, qt: QuadTree, theta: float):
    """
    Computes negative forces in barnes-hut algorithm.

    :param y: Input data
    :param qt: Quad tree constructed on input data
    :param theta: Threshold for summarizing data
    :return: calculated negative forces
    """

    n_sample = y.shape[0]
    n_dimensions = 2
    negative_forces = np.zeros((n_sample, n_dimensions))
    sum_Q = 0

    # for each input point
    for i in range(n_sample):
        # take point ot query
        point = y[i]
        # get summary of query point from quad tree
        results = qt.summarize(point, theta*theta)
        # calculate qijZ
        qijZ = 1.0 / (1.0 + results[:, 2])
        # calculate intermediate results
        NqijZ = qijZ * results[:, 3]
        sum_Q += np.sum(NqijZ)
        mult = NqijZ * qijZ
        # calculate negative forces
        negative_forces[i, 0] = np.sum(mult * results[:, 0])
        negative_forces[i, 1] = np.sum(mult * results[:, 1])

    sum_Q = max(sum_Q, FLOAT64_EPS)
    return negative_forces, sum_Q


def positive_forces_gradient(P: np.array, y: np.array, neighbors_idx: np.array, indptr_P, sum_Q: float):
    """
    Computes positive forces in barnes-hut algorithm.

    :param P: Pairwise joint probability of high dimension input data
    :param y: Input data
    :param neighbors_idx: Neighbourhood y point indices
    :param indptr_P: Index of non 0 values
    :param sum_Q: Denominator
    :return: calculated positive forces
    """

    n_dimensions = y.shape[1]
    n_sample = indptr_P.shape[0] - 1
    positive_forces = np.zeros((n_sample, n_dimensions))
    error = 0

    # for each input point
    for i in range(n_sample):
        # get p_ij values for current point
        P_neighbors = P[indptr_P[i]: indptr_P[i+1]]
        # get y_points for multiplying with P points
        y_neighbors = y[neighbors_idx[indptr_P[i]: indptr_P[i+1]]]
        # calculate q_ij for y_i
        residual = y[i] - y_neighbors 
        q_ij = 1.0 / (1.0 + np.sum(residual**2, axis = 1))
        PQ = P_neighbors * q_ij
        # calculate positive forces
        positive_forces[i, 0] = np.sum(PQ * residual[:, 0])
        positive_forces[i, 1] = np.sum(PQ * residual[:, 1])

        q_ij_ = q_ij/sum_Q
        # calculate KL divergence
        error += np.sum(P_neighbors * np.log(np.maximum(P_neighbors, FLOAT32_TINY)/ np.maximum(q_ij_, FLOAT32_TINY)))

    return positive_forces, error
