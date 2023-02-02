# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 18:30:04 2020

@author: Filip
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time


class NeighborsHeap:
    
    def __init__(self, n_points, n_neighbors):
        self.distances = np.inf + np.zeros((n_points, n_neighbors))
        self.indices = np.zeros((n_points, n_neighbors), dtype = np.int64)

    def maximum_distance(self, row):
        return self.distances[row, 0]
    
    def get_arrays(self, sort = True):
        if sort:
            self._sort()
        return self.distances, self.indices
    
    def push(self, row, val, i_val):
        
        size = self.distances.shape[1]
        dist_arr = self.distances[row]
        ind_arr = self.indices[row]

        # check if val should be in heap
        if val > dist_arr[0]:
            return 0

        # insert val at position zero
        dist_arr[0] = val
        ind_arr[0] = i_val

        #descend the heap, swapping values until the max heap criterion is met
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
        return 0
    
    def _sort(self):
        for row in np.arange(0, self.distances.shape[0]):
            sorted_idxs = np.argsort(self.distances[row])
            self.distances[row] = self.distances[row, sorted_idxs]
            self.indices[row] = self.indices[row, sorted_idxs]
        return 0
    
    def largest(self, point_index):
        return self.distances[point_index, 0]

        
def swap(arr, ind1, ind2):
        tmp = arr[ind1]
        arr[ind1] = arr[ind2]
        arr[ind2] = tmp
        return arr   


class KDTree:
    
    # Node of kd -tree
    class Node:    
    
        def __init__(self, idx_start=0, idx_end=0, is_leaf=False, radius = 0, feature_split = 0, index_split = 0):
            self.idx_start = idx_start
            self.idx_end = idx_end
            self.is_leaf = is_leaf
            self.radius = radius
            self.feature_split = feature_split
            self.index_split = index_split


    # Distance metric
    class dist_metric:
        
        def __init__(self, metric= 'minkowski', p=2):
            self.metric = metric
            self.p = p
           
            
    def __init__(self, data, leaf_size=40):
        self.data = np.asarray(data)

        self.n_samples, self.n_features = np.shape(self.data)
        self.leaf_size = int(leaf_size)
        self.dist_metric = self.dist_metric('euclidian', 2)
         
        if self.leaf_size < 1:
            raise ValueError("leafsize must be at least 1")
        
        # set number of levels of kd-tree
        self.n_levels = int(np.log2(max(1, (self.n_samples - 1) / self.leaf_size))) + 1
        # set number of nodes of kd-tree
        self.n_nodes = (2 ** self.n_levels) - 1
      
        # allocate index array and node data
        self.idx_array = np.arange(self.n_samples)
        # init node data
        self.node_data = [self.Node() for i in range(self.n_nodes)]

        # allocate node boundes
        self.node_bounds = np.zeros((2, self.n_nodes, self.n_features))       
        self.__build(0, 0, self.n_samples)
    
    
    def init_new_node(self, i_node, idx_start, idx_end):

        rad = 0
        # find lower ad upper bounds of each feature
        self.node_bounds[0, i_node, :] = np.amin(self.data[self.idx_array[idx_start: idx_end], :], axis = 0)
        self.node_bounds[1, i_node, :] = np.amax(self.data[self.idx_array[idx_start: idx_end], :], axis = 0)
        
        # calculate radius
        rad = np.sum(pow(0.5 * abs(self.node_bounds[1, i_node, :] - self.node_bounds[0, i_node, :]), self.dist_metric.p))
        
        # initiate node data
        self.node_data[i_node].idx_start = idx_start
        self.node_data[i_node].idx_end  = idx_end 
        self.node_data[i_node].radius = pow(rad, 1. / self.dist_metric.p)
        
        return 0
    
    
    def __build(self, i_node, idx_start, idx_end):

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
            j_max = self.find_node_split_feature(self.data, node_idx_array, self.n_features)
            
            # divide indices by meadian in 2 groups            
            self.partition_node_indices(self.data, node_idx_array, j_max, n_mid)
        
            self.node_data[i_node].feature_split = j_max
            self.node_data[i_node].index_split = node_idx_array[n_mid]
            
            # build left and right subtree
            self.__build(2 * i_node + 1, idx_start, idx_start + n_mid)
            self.__build(2 * i_node + 2, idx_start + n_mid, idx_end)
        return 0
    
    def find_node_split_feature(self, data, node_indices, n_features):
        '''
        return index of feature with max spread:
        j_max = np.argmax(data[node_indices, j].max() - data[node_indices, j].min())
        '''
        j_max = 0
        max_spread = 0
        
        for j in np.arange(n_features):
            max_feature_val = np.amax(data[node_indices, j])
            min_feature_val = np.amin(data[node_indices, j])
            spread = max_feature_val - min_feature_val
            if spread > max_spread:
                max_spread = spread
                j_max = j
                
        return j_max

    
    def partition_node_indices(self, data, node_indices, split_dim, split_index):
        ''' partly implementation of quick sort 
            return  partly sorted node_indices - > 
                data[node_indices[:split_index], split_dim] < data[node_indices[split_index], split_dim]
                data[node_indices[split_index:node_indices[-1]], split_dim] > data[node_indices[split_index], split_dim]
        '''
        left = 0
        right = node_indices.shape[0] - 1
    
        while True:
            midindex = left
            for i in range(left, right):
                d1 = data[node_indices[i], split_dim]
                d2 = data[node_indices[right], split_dim]
                if d1 < d2:
                    # swap
                    swap(node_indices, i, midindex)
                    midindex += 1
            #swap
            swap(node_indices, midindex, right)
            if midindex == split_index:
                break
            elif midindex < split_index:
                left = midindex + 1
            else:
                right = midindex - 1
    
        return 0    
    
    def rdist_to_dist(self, distances):
        return pow(distances, 1. / self.dist_metric.p)
    
    def query(self, X, k=1, sort_results=True):
        self.n_trims = 0
        self.n_leaves = 0
        self.n_splits = 0

        if len(X.shape) > 2:
            raise ValueError("query data0 must be 2D")
           
        if X.shape[1] != self.data.shape[1]:
            raise ValueError("query data dimension must match training data dimension")
            
        if self.data.shape[0] < k:
            raise ValueError("k must be less than or equal to the number of training points")
        
        # init heat neighbours            
        heap = NeighborsHeap(X.shape[0], k)

        for i in range(X.shape[0]):
            point = X[i,:]
            reduced_dist_LB = self.min_rdist(0, point)
            self.query_dfs(0, point, i, heap, reduced_dist_LB)

        distances, indices = heap.get_arrays(sort=sort_results)
        distances = self.rdist_to_dist(distances)
        return distances, indices
    
    def rdist(self, point, points):
        return np.sum(pow(points - point, self.dist_metric.p), axis = 1)

    def query_dfs(self, i_node, point, index_point, heap, reduced_dst_from_bounds):
        node_i = self.node_data[i_node]
        
        # node is outise of node radius - > trim if from the query
        if reduced_dst_from_bounds > heap.largest(index_point):
        #dis, ind = heap.get_arrays(sort= False)
        #if reduced_dst_from_bounds > dis[index_point,0]:
            self.n_trims += 1
        
        # node is leaf -> update heap of nearby points
        elif node_i.is_leaf:
            self.n_leaves += 1
            data_in_node = self.data[self.idx_array[node_i.idx_start:node_i.idx_end], :]
            # calculate distances from point to nearby points
            dist_from_point = self.rdist(point, data_in_node)
            # if distance < leargest distance in priority queue push neightbor distance, index
            for i, node_data_index in enumerate(range(node_i.idx_start, node_i.idx_end)):
                if dist_from_point[i] < heap.largest(index_point):
                #dis, ind = heap.get_arrays(sort= False)
                #if dist_from_point[i] < dis[index_point,0]:
                    heap.push(index_point, dist_from_point[i], self.idx_array[node_data_index])
                    
        
        # node is not leaf -> query subnodes
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
                
        return 0
    
    def min_rdist(self, i_node, point):
        ''' 
        if point is inside node area -> distance = 0, else calculate distance(point, area)**p
        '''
        d_lo = self.node_bounds[0, i_node, :] - point
        d_hi = point - self.node_bounds[1, i_node, :]
        d = (d_lo + abs(d_lo)) + (d_hi + abs(d_hi))
        rdist = np.sum(pow(0.5 * d, self.dist_metric.p))
        return rdist
        
    def min_dist(self, i_node, point):
        return pow(self.min_rdist(i_node, point), 1. / self.dist_metric.p)
    
    def visualize_bounds(self):
        if self.data.shape[1] > 2:
            raise ValueError("Bounds can be visualize just for 2 D data")
        plt.plot(self.data[:,0], self.data[:,1], 'ro')
        plt.title("K-d stablo")
        plt.xlabel("x1")
        plt.ylabel("x2")
        for i_node in np.arange(0, self.node_bounds.shape[1]):
            i_node_lower_bounds = self.node_bounds[0, i_node, :]
            i_node_upper_bounds = self.node_bounds[1, i_node, :]

            if self.node_data[i_node].is_leaf:          
                plt.plot([i_node_lower_bounds[0], i_node_upper_bounds[0], i_node_upper_bounds[0], i_node_lower_bounds[0], 
                          i_node_lower_bounds[0]], 
                        [i_node_lower_bounds[1], i_node_lower_bounds[1], i_node_upper_bounds[1], i_node_upper_bounds[1], 
                         i_node_lower_bounds[1]])
        plt.show()
        

def visualize_tree_bounds(data, node_bounds, node_data):
        if data.shape[1] > 2:
            raise ValueError("Bounds can be visualize just for 2 D data")
        plt.plot(data[:,0], data[:,1], 'ro')
        plt.title("KD-tree -scikitlearn")
        plt.xlabel("x1")
        plt.ylabel("x2")
        for i_node in np.arange(0, node_bounds.shape[1]):
            i_node_lower_bounds = node_bounds[0, i_node, :]
            i_node_upper_bounds = node_bounds[1, i_node, :]
            if node_data['is_leaf'][i_node]:          
                plt.plot([i_node_lower_bounds[0], i_node_upper_bounds[0], i_node_upper_bounds[0], i_node_lower_bounds[0], 
                          i_node_lower_bounds[0]], 
                        [i_node_lower_bounds[1], i_node_lower_bounds[1], i_node_upper_bounds[1], i_node_upper_bounds[1], 
                         i_node_lower_bounds[1]])
        plt.show()
  

    