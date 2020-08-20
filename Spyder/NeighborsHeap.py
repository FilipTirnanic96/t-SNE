# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 20:49:23 2020

@author: Filip
"""
import numpy as np
from time import time

class NeighborsHeap:
    
    def __init__(self, n_points, n_neighbors):
        self.distances = np.inf + np.zeros((n_points, n_neighbors))
        self.indices = np.zeros((n_points, n_neighbors), dtype = np.int32)
        
        
    def maximum_distance(self, row):
        return self.distances[row, 0]
    
    def get_arrays(self, sort = True):
        if sort:
            self._sort()
        return self.distances, self.indices
    
    def push(self, row, val, i_val):
        
        size = self.distances.shape[1]
        dist_arr = self.distances[row, :]
        ind_arr = self.indices[row, :]

        # check if val should be in heap
        if val > dist_arr[0]:
            return 0

        # insert val at position zero
        dist_arr[0] = val
        ind_arr[0] = i_val
        index_child_1 = 0
        index_child_2 = 0
        i_swap = 0
        #descend the heap, swapping values until the max heap criterion is met
        i = np.int32(0)
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
            sorted_idxs = np.argsort(self.distances[row, :])
            self.distances[row, :] = self.distances[row, sorted_idxs]
            self.indices[row, :] = self.indices[row, sorted_idxs]
        return 0
    
    def largest(self, point_index):
        return self.distances[point_index, 0]
    
from sklearn.neighbors import kd_tree    
if __name__ == "__main__":
    distance = np.arange(0,10000)
    indeces = np.arange(0,distance.shape[0])
    nn_heap = NeighborsHeap(distance.shape[0], distance.shape[0])
    nn_heap1 = kd_tree.NeighborsHeap(distance.shape[0], distance.shape[0])
    start = time() 
    for val, i_val in zip(distance,indeces):
       nn_heap.push(0, val, i_val) 
    end1 = time() - start 
    
    start = time() 
    for val, i_val in zip(distance,indeces):
       nn_heap1.push(0, val, i_val) 
    end2 = time() - start 
    dist, ind = nn_heap.get_arrays()