# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 18:30:04 2020

@author: Filip
"""

import numpy as np
import matplotlib.pyplot as plt


class NeighborsHeap:

    def __init__(self, n_points: int, n_neighbors: int):
        self.distances = np.inf + np.zeros((n_points, n_neighbors))
        self.indices = np.zeros((n_points, n_neighbors), dtype=np.int64)

    def maximum_distance(self, row: int):
        """
        Get maximum neighbor distance from point at index row

        :param row: Point index
        :return: Maximum distance from point to ints neighbours
        """

        return self.distances[row, 0]

    def get_arrays(self, sort: bool = True):
        """
        Get distances and indices array.

        :param sort: Flag if array should be sorted
        :return: Distances and indices arrays
        """

        if sort:
            self._sort()
        return self.distances, self.indices

    def push(self, row: int, val: int, i_val: int):
        """
        Push new neighbour point with its distance from row point.

        :param row: Point index
        :param val: Distance of point to its neighbour point
        :param i_val: Neighbour point index
        """

        size = self.distances.shape[1]
        dist_arr = self.distances[row]
        ind_arr = self.indices[row]

        # check if val should be in heap
        if val > dist_arr[0]:
            return 0

        # insert val at position zero
        dist_arr[0] = val
        ind_arr[0] = i_val

        # descend the heap, swapping values until the max heap criterion is met
        i = 0
        while True:
            index_child_1 = 2 * i + 1
            index_child_2 = index_child_1 + 1

            if index_child_1 >= size:
                break
            elif index_child_2 >= size:
                if val < dist_arr[index_child_1]:
                    i_swap = index_child_1
                else:
                    break
            elif dist_arr[index_child_1] >= dist_arr[index_child_2]:
                if val < dist_arr[index_child_1]:
                    i_swap = index_child_1
                else:
                    break
            else:
                if val < dist_arr[index_child_2]:
                    i_swap = index_child_2
                else:
                    break

            dist_arr[i] = dist_arr[i_swap]
            ind_arr[i] = ind_arr[i_swap]

            i = i_swap

        dist_arr[i] = val
        ind_arr[i] = i_val

    def _sort(self):
        """
        Sort distances and indices array.
        """

        for row in np.arange(0, self.distances.shape[0]):
            sorted_idxs = np.argsort(self.distances[row])
            self.distances[row] = self.distances[row, sorted_idxs]
            self.indices[row] = self.indices[row, sorted_idxs]

    def largest(self, point_index: int):
        """
        Return larges distance from current point in max heap.

        :param point_index: Data point
        """

        return self.distances[point_index, 0]


def swap(arr: np.array, ind1: int, ind2: int):
    """
    Performs a swap in array.

    :param arr: Data array
    :param ind1: Index1 to be swapped
    :param ind2: Index2 to be swapped
    """

    tmp = arr[ind1]
    arr[ind1] = arr[ind2]
    arr[ind2] = tmp
    return arr


class KDTree:

    class Node:

        def __init__(self, idx_start: int = 0, idx_end: int = 0, is_leaf: bool = False, radius: float = 0, feature_split: int = 0, index_split: int = 0):
            self.idx_start = idx_start
            self.idx_end = idx_end
            self.is_leaf = is_leaf
            self.radius = radius
            self.feature_split = feature_split
            self.index_split = index_split

    def __init__(self, data: np.array, leaf_size: int = 40):
        self.data = np.asarray(data)

        self.n_samples, self.n_features = np.shape(self.data)
        self.leaf_size = int(leaf_size)

        if self.leaf_size < 1:
            raise ValueError("Leaf size must be at least 1")

        # set number of levels of kd-tree
        self.n_levels = int(np.log2(max(1, (self.n_samples - 1) / self.leaf_size))) + 1
        # set number of nodes of kd-tree
        self.n_nodes = (2 ** self.n_levels) - 1

        # allocate index array and node data
        self.idx_array = np.arange(self.n_samples)
        # init node data
        self.node_data = [self.Node() for i in range(self.n_nodes)]

        # allocate node bounds
        self.node_bounds = np.zeros((2, self.n_nodes, self.n_features))

        # init kd-tree parameters
        self.n_trims = 0
        self.n_leaves = 0
        self.n_splits = 0

        # build kd-tree
        self.__build(0, 0, self.n_samples)

    def init_new_node(self, i_node: int, idx_start: int, idx_end: int):
        """
        Create new node.

        :param i_node: Index of current node
        :param idx_start: Index start of data in i_node
        :param idx_end: Index end of data in i_node
        """

        # find lower ad upper bounds of each feature
        self.node_bounds[0, i_node, :] = np.amin(self.data[self.idx_array[idx_start: idx_end], :], axis=0)
        self.node_bounds[1, i_node, :] = np.amax(self.data[self.idx_array[idx_start: idx_end], :], axis=0)

        # calculate radius
        rad = np.sum(pow(0.5 * abs(self.node_bounds[1, i_node, :] - self.node_bounds[0, i_node, :]), 2.))

        # initiate node data
        self.node_data[i_node].idx_start = idx_start
        self.node_data[i_node].idx_end = idx_end
        self.node_data[i_node].radius = pow(rad, 1. / 2.)

    def __build(self, i_node: int, idx_start: int, idx_end: int):
        """
        Create kd-tree from input data.

        :param i_node: Index of current node
        :param idx_start: Index start of data in i_node
        :param idx_end: Index end of data in i_node
        """

        n_points = idx_end - idx_start
        n_mid = int(n_points / 2)
        node_idx_array = self.idx_array[idx_start: idx_end]

        # init new node
        self.init_new_node(i_node, idx_start, idx_end)

        if 2 * i_node + 1 >= self.n_nodes:
            self.node_data[i_node].is_leaf = True

        elif idx_end - idx_start < 2:
            self.node_data[i_node].is_leaf = True

        else:
            # find feature for split
            self.node_data[i_node].is_leaf = False
            j_max = self.find_node_split_feature(node_idx_array)

            # divide indices by median in 2 groups
            self.partition_node_indices(node_idx_array, j_max, n_mid)

            self.node_data[i_node].feature_split = j_max
            self.node_data[i_node].index_split = node_idx_array[n_mid]

            # build left and right subtree
            self.__build(2 * i_node + 1, idx_start, idx_start + n_mid)
            self.__build(2 * i_node + 2, idx_start + n_mid, idx_end)

    def find_node_split_feature(self, node_indices: np.array):
        """
        Find index of feature with max spread in current node data.

        :param node_indices: Indices of data in current node
        :return j_max: Index of feature to split data in current node
        """

        j_max = 0
        max_spread = 0

        # iterate over each feature
        for j in np.arange(self.n_features):
            # get max and min values of feature in current node data
            max_feature_val = np.amax(self.data[node_indices, j])
            min_feature_val = np.amin(self.data[node_indices, j])
            # calculate max spread of current feature
            spread = max_feature_val - min_feature_val
            if spread > max_spread:
                max_spread = spread
                j_max = j

        return j_max

    def partition_node_indices(self, node_indices: np.array, split_dim: int, split_index: int):
        """
        Partly implementation of quick sort. Sorts all data so on left side of split index data
        is has less value and on right side data has greater value in current split feature

        :param node_indices: Indices of data in current node
        :param split_dim: Feature for which we perform split
        :param split_index: Mid point index
        """

        left = 0
        right = node_indices.shape[0] - 1

        while True:
            midindex = left

            for i in range(left, right):

                d1 = self.data[node_indices[i], split_dim]
                d2 = self.data[node_indices[right], split_dim]
                if d1 < d2:
                    swap(node_indices, i, midindex)
                    midindex += 1

            swap(node_indices, midindex, right)
            if midindex == split_index:
                break
            elif midindex < split_index:
                left = midindex + 1
            else:
                right = midindex - 1

    def query(self, X: np.array, k: int = 1, sort_results: bool = True):
        """
        Query kd tree. Returns distances and indices of neighbours for all query points.

        :param X: Query data
        :param k: Number of neighbours
        :param sort_results: Flag if results should be sorted
        :return: Distances and indices of neighbours for all query points
        """

        self.n_trims = 0
        self.n_leaves = 0
        self.n_splits = 0

        if X.shape[1] != self.data.shape[1]:
            raise ValueError("query data dimension must match training data dimension")

        if self.data.shape[0] < k:
            raise ValueError("k must be less than or equal to the number of training points")

        # init heat neighbours            
        heap = NeighborsHeap(X.shape[0], k)

        # for each point perform query
        for i in range(X.shape[0]):
            point = X[i, :]
            reduced_dist_LB = self.min_rdist(0, point)
            self.query_dfs(0, point, i, heap, reduced_dist_LB)

        distances, indices = heap.get_arrays(sort=sort_results)
        distances = pow(distances, 1. / 2.)
        return distances, indices

    def rdist(self, point: np.array, points: np.array):
        """
        Calculate euclidean distance from point to points.

        :param point: Data point
        :param points: Data points
        """

        return np.sum(pow(points - point, 2.), axis=1)

    def query_dfs(self, i_node, point, index_point, heap, reduced_dst_from_bounds):
        """
        Perform depth first search query. Fill heap with all relevant data for query point.

        :param i_node: Index of current node
        :param point: Query data point
        :param index_point: Index of query data point
        :param reduced_dst_from_bounds: Distance of point to node area
        :param heap: Data points

        """
        node_i = self.node_data[i_node]

        # node is outise of node radius - > trim if from the query
        if reduced_dst_from_bounds > heap.largest(index_point):
            self.n_trims += 1

        # node is leaf -> update heap of nearby points
        elif node_i.is_leaf:
            self.n_leaves += 1
            data_in_node = self.data[self.idx_array[node_i.idx_start:node_i.idx_end], :]
            # calculate distances from point to nearby points
            dist_from_point = self.rdist(point, data_in_node)
            # if distance < leargest distance in priority queue push neighbor distance, index
            for i, node_data_index in enumerate(range(node_i.idx_start, node_i.idx_end)):
                if dist_from_point[i] < heap.largest(index_point):
                    heap.push(index_point, dist_from_point[i], self.idx_array[node_data_index])
        # node is not leaf -> query sub nodes
        else:
            self.n_splits += 1
            index_child_1 = 2 * i_node + 1
            index_child_2 = index_child_1 + 1

            reduced_dst_from_bounds_1 = self.min_rdist(index_child_1, point)
            reduced_dst_from_bounds_2 = self.min_rdist(index_child_2, point)

            # point is closer to child 1 recursively query subtree of child 1
            if reduced_dst_from_bounds_1 <= reduced_dst_from_bounds_2:
                self.query_dfs(index_child_1, point, index_point, heap, reduced_dst_from_bounds_1)
                self.query_dfs(index_child_2, point, index_point, heap, reduced_dst_from_bounds_2)
            # point is closer to child 1 recursively query subtree of child 2
            else:
                self.query_dfs(index_child_2, point, index_point, heap, reduced_dst_from_bounds_2)
                self.query_dfs(index_child_1, point, index_point, heap, reduced_dst_from_bounds_1)

    def min_rdist(self, i_node, point):
        """
        Calculate euclidean distance from point to node area.

        :param i_node: Index of current node
        :param point: Data point
        :return rdist: Distance of point to node area
        """

        d_lo = self.node_bounds[0, i_node, :] - point
        d_hi = point - self.node_bounds[1, i_node, :]
        # calculate distance
        d = (d_lo + abs(d_lo)) + (d_hi + abs(d_hi))
        rdist = np.sum(pow(0.5 * d, 2.))

        return rdist

    def visualize_bounds(self):
        """
        Visualise bounds of quad tree.
        """

        if self.data.shape[1] > 2:
            raise ValueError("Bounds can be visualize just for 2D data")

        plt.plot(self.data[:, 0], self.data[:, 1], 'ro')
        plt.title("Kd tree")
        plt.xlabel("x1")
        plt.ylabel("x2")
        for i_node in np.arange(0, self.node_bounds.shape[1]):
            i_node_lower_bounds = self.node_bounds[0, i_node, :]
            i_node_upper_bounds = self.node_bounds[1, i_node, :]

            if self.node_data[i_node].is_leaf:
                plt.plot(
                    [i_node_lower_bounds[0], i_node_upper_bounds[0], i_node_upper_bounds[0], i_node_lower_bounds[0],
                     i_node_lower_bounds[0]],
                    [i_node_lower_bounds[1], i_node_lower_bounds[1], i_node_upper_bounds[1], i_node_upper_bounds[1],
                     i_node_lower_bounds[1]])
        plt.show()
