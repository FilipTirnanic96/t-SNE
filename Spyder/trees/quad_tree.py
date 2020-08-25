# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:31:26 2020

@author: Filip
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time

class QuadTree:
    
    
    class Cell:
        
        def __init__(self, parent_id, cell_id, depth):
            self.parent_id = parent_id
            self.cell_id = cell_id
            self.is_leaf = True
            self.depth = depth
            self.squared_max_width = 0
            self.cumulative_size = 0
            self.center = np.zeros(2)
            self.children = [None, None, None, None]
            self.min_bounds = np.zeros(2)
            self.max_bounds = np.zeros(2)
            self.masscenter = np.zeros(2)
            self.point_index = 0
            
    def __init__(self):       
        self.n_dimensions = 2
        self.n_cells_per_cell = self.n_dimensions**2
        self.cell_count = 0
        self.n_points = 0
        self.cells = []     
        self.eps = 1e-6
        
    def build_tree(self, X):
        
        n_samples = X.shape[0]
        minX = np.min(X, axis=0)
        maxX = np.max(X, axis=0)
        maxX = np.maximum(maxX * (1. + 1e-3 * np.sign(maxX)), maxX + 1e-3)
        self.init_root(minX, maxX)
        
        for i in range(n_samples):
            point = X[i]
            self.insert_point(point, i, 0)
            
        
    def init_root(self, min_bounds, max_bounds):
        
        root_cell = self.Cell(-1, 0, 0)        
        root_cell.min_bounds = min_bounds
        root_cell.max_bounds = max_bounds
        root_cell.center = (max_bounds + min_bounds) / 2.
        width = max_bounds - min_bounds
        root_cell.squared_max_width = np.amax(width**2)
        root_cell.cell_id = 0

        self.cells.append(root_cell)    
        self.cell_count += 1
        return 0
        
    
    
    def insert_point(self, point, point_index, cell_id):

        cell = self.cells[cell_id]
        cell_n_points = cell.cumulative_size
        # if cell is empthy leaf insert point in cell
        if cell.cumulative_size == 0:
            
            cell.cumulative_size += 1 
            self.n_points += 1
            cell.masscenter = point
            cell.point_index = point_index
            return cell_id
        
        # if current cell isn't leaf update cell centar and recurse in cell child
        if not cell.is_leaf:
            
            # update masscenter cell mean by iterative mean update
            cell.masscenter = (cell_n_points * cell.masscenter + point) / (cell_n_points + 1)
            
            # update cumulative cell size
            cell.cumulative_size += 1
            selected_child = self._select_child(point, cell)            
            if selected_child == None:
                self.n_points += 1
                return self._insert_point_in_new_child(point, point_index, cell)
            
            return self.insert_point(point, point_index, selected_child)
        
        # if cell is leaf and point is already inserted update num of point in that cell
        if self._is_duplicate(cell.masscenter, point):
            cell.cumulative_size+=1
            self.n_points +=1
            return cell_id
        
        # In a leaf, the masscenter correspond to the only point included in current leaf cell.
        # So push thah point in the child cell and then inser new point in childs
        self._insert_point_in_new_child(cell.masscenter, cell.point_index, cell)
        
        return self.insert_point(point, point_index, cell_id)
        
    def _select_child(self, point, cell):
        selected_child_id = 0
                     
        for i in range(self.n_dimensions):
            selected_child_id *=2
            if point[i] >= cell.center[i]:
                selected_child_id += 1
        
        return cell.children[selected_child_id]
    
        
    def _is_duplicate(self, point1, point2):
        return (abs(point1[0] - point2[0]) < self.eps and abs(point1[1] - point2[1]) < self.eps)
     
       
    def _insert_point_in_new_child(self, point, point_index, cell): 
        cell_id = self.cell_count
        self.cell_count += 1
        # init new child
        child = self.Cell(cell.cell_id, cell_id, cell.depth + 1)
        self.cells.append(child)
        
        # find chid id (quadrant of a child)
        child_cell_id = 0
        for i in range(self.n_dimensions):
            child_cell_id *= 2
            if point[i] >= cell.center[i]:
                child_cell_id += 1
                child.min_bounds[i] = cell.center[i]
                child.max_bounds[i] = cell.max_bounds[i]
            else:
                child.min_bounds[i] = cell.min_bounds[i]
                child.max_bounds[i] = cell.center[i]
        
        # set center of child quadrant
        child.center = (child.min_bounds + child.max_bounds) / 2.
        # calculate width of child quadran
        width = child.max_bounds - child.min_bounds
        
        # set center of the mass of cell
        child.masscenter = point
        child.squared_max_width = np.amax(width**2)
        
        
        # set prent is_leaf to False
        cell.is_leaf = False
        cell.point_index = -1
        
        
        child.point_index = point_index
        child.cumulative_size = 1
        
        # Store the child cell in the correct place in children
        cell.children[child_cell_id] = child.cell_id

        return child.cell_id    
    
    def get_cell(self, point, cell_id = 0):
        cur_cell = self.cells[cell_id]
        if cur_cell.is_leaf:
            if self._is_duplicate(cur_cell.masscenter, point):
                return cell_id
            else:
                raise ValueError("Query point not in the Tree.")
                
        selected_child = self._select_child(point, cur_cell)
        if selected_child != None:            
            return self.get_cell(point, selected_child)
            
        else:
            raise ValueError("Query point not in the Tree.")

    def summarize(self, point, squared_theta):
        results = np.zeros((self.n_points, self.n_dimensions + 2))
        index = self._summarize(point, results, squared_theta) 
        return results[0:index, :]

    def _summarize(self, point, results, squared_theta = 0.5, cell_id = 0, index = 0):
        cur_cell = self.cells[cell_id]
        # calculate point_i - cell_points_mean
        results[index, 0:2] = point - cur_cell.masscenter
        # get eulidian distance euc(point_i - cell_points_mean)
        results[index,2] = np.sum(results[index, 0:2]**2)

        is_duplicate = (np.abs(results[index, 0]) < self.eps and np.abs(results[index, 1]) < self.eps)
        
        # if we find the same point in leaf cell return
        if is_duplicate and cur_cell.is_leaf:
            return index
        
        # Check if we can use current node as summary of points in its sub tree
        # Compare distances of point to the diagoanl of cell -> if it is less then theta**2 sub tree cells can be summerize with this cell
        # else recursive call other childs of current cell
        if cur_cell.is_leaf or ((cur_cell.squared_max_width / results[index, 2]) < squared_theta):
            results[index,3] = cur_cell.cumulative_size
            return index + 1
        else:
            for child in range(self.n_cells_per_cell):
                child_id = cur_cell.children[child]
                if child_id != None:
                    index = self._summarize(point, results, squared_theta, child_id, index)
        
        return index
     
    def visualize_bounds(self):
        plt.title("Quad-tree")
        plt.xlabel("x1")
        plt.xlabel("x2")
        for cell in self.cells:
            i_node_lower_bounds = cell.min_bounds
            i_node_upper_bounds = cell.max_bounds

            if cell.is_leaf:          
                plt.plot([i_node_lower_bounds[0], i_node_upper_bounds[0], i_node_upper_bounds[0], i_node_lower_bounds[0], 
                          i_node_lower_bounds[0]], 
                        [i_node_lower_bounds[1], i_node_lower_bounds[1], i_node_upper_bounds[1], i_node_upper_bounds[1], 
                         i_node_lower_bounds[1]])
            else:
                plt.plot([i_node_lower_bounds[0], i_node_upper_bounds[0], i_node_upper_bounds[0], i_node_lower_bounds[0], 
                          i_node_lower_bounds[0]], 
                        [i_node_lower_bounds[1], i_node_lower_bounds[1], i_node_upper_bounds[1], i_node_upper_bounds[1], 
                         i_node_lower_bounds[1]], 'k')              
        plt.show()
    
    def get_cumulative_size_list(self):
        return [cell.cumulative_size for cell in self.cells]
    
    def get_leaf_list(self):
        return [cell.is_leaf for cell in self.cells]
    
if __name__ == "__main__":   
    import quad_tree1     
    from sklearn.neighbors._quad_tree import _QuadTree
    np.random.seed(50)
    num_class = 20
    mean = np.array([1,1])
    cov =  np.array([[1, 0.2],[0.2, 1]])
    X = np.random.multivariate_normal(mean, cov, num_class).astype(np.float32)           
    
    plt.plot(X[:,0], X[:,1], 'ro')   
    
    qt = QuadTree()
    
    start = time()
    qt.build_tree(X)
    end1 = time() - start   
    
    qt.visualize_bounds()       
    qt1 = _QuadTree(X.shape[1],verbose = 1)
    
    start = time()
    qt1.build_tree(X) 
    end2 = time() - start 
    
    start = time()
    results = qt.summarize(X[0,:], 0.25)  
    end3 = time() - start
    
    theta = 0.5     
    point = X[0] 
    
    start = time()
    idx, summary = quad_tree1._py_summarize(qt1, point, X, theta)
    end4 = time() - start
    
    summary = summary[0:idx]
    sumary = summary.reshape((-1, 4))     
    
    c_list = qt.get_cumulative_size_list()
    c_list1 = qt1.cumulative_size
    
    l_list = qt.get_leaf_list()
    l_list1 = qt1.leafs