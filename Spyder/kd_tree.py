# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 18:30:04 2020

@author: Filip
"""

import numpy as np

class Node():
    
    # Init node
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
        
        
    # returns true is node is leaf; else returns false
    def is_leaf(self):
        return (self.left == None and self.right == None)         
        
    
    # 0 - left; other value = right
    def set_child(self, child_node, pos):
        if( pos == 0 ):
            self.left = child_node
        else:
            self.right = child_node
    
    # return children
    def get_children(self):
        return self.left, self.right
    
class KDTree():
 
    def __init__(self, data, leafsize=10):
        self.data = np.asarray(data)

        self.n, self.m = np.shape(self.data)
        self.leafsize = int(leafsize)
        if self.leafsize < 1:
            raise ValueError("leafsize must be at least 1")
        self.maxes = np.amax(self.data,axis=0)
        self.mins = np.amin(self.data,axis=0)

        self.tree = self.__build(np.arange(self.n), self.maxes, self.mins)
   
    
    
    
    
    
    
def height(node):
    if node == None:
        return 0
    left_height = height(node.left)
    right_height = height(node.right)
    
    if(left_height > right_height):
        return left_height + 1
    else:
        return right_height + 1
    
def preorder(node):
    if node == None:
        return
    print(node.data)
    preorder(node.left)
    preorder(node.right)
    
def inorder(node):
    if node == None:
        return
    inorder(node.left)
    print(node.data)
    inorder(node.right)

def postorder(node):
    if node == None:
        return
    postorder(node.left)
    postorder(node.right)  
    print(node.data)




if __name__ == "__main__":
    '''root_node = Node(data = 1)
    root_node.set_child(Node(data = 2), 0)
    root_node.set_child(Node(data = 3), 1)
    root_node_left, root_node_right = root_node.get_children()
    root_node_left.set_child(Node(data = 4), 0)
    root_node_left.set_child(Node(data = 5), 1)
    root_node_right.set_child(Node(data = 6), 0)
    root_node_right.set_child(Node(data = 7), 1)'''
    #preorder(root_node)
    #postorder(root_node)
    #print(height(root_node))

    num_class = 20
    mean = np.array([1,1])
    cov =  np.array([[1, 0.2],[0.2, 1]])
    X = np.random.multivariate_normal(mean, cov, num_class)
    #y = np.zeros((num_class))


    #tree = create(points)
    