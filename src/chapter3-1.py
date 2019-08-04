import numpy as np
from collections import namedtuple
from math import sqrt

class KdNode(object):
    def __init__(self, elt, split, left, right):
        self.elt = elt
        self.split = split
        self.left = left
        self.right = right

class KdTree(object):
    def __init__(self, data):
        k = len(data[0]) # 数据维度
        
        def createNode(split, dataset):#按split维划分
            if len(dataset) == 0:
                return None
            dataset.sort(key = lambda x : x[split]) #按split维度进行排序，默认为升序
            split_pos = len(dataset)//2
            mid_elt = dataset[split_pos]
            split_next = (split + 1) % k
            return KdNode(mid_elt, split, createNode(split_next, dataset[:split_pos]), createNode(split_next, dataset[split_pos + 1 :]))
        
        self.root = createNode(0, data)

def preOrder(root):
    print(root.elt)
    if root.left:
        preOrder(root.left)
    if root.right:
        preOrder(root.right)
        
result = namedtuple("Result_tuple", "nearest_point nearest_dist visited_nodes")
def find_nearest(tree, point_elt):
    k = len(point_elt)
    def travel(kd_node, target_elt, max_dist):
        if kd_node is None:
            return result([0]*k, float("inf"), 0)
        
        visited_nodes = 1
        s = kd_node.split
        pivot = kd_node.elt
        
        if target_elt[s] <= pivot[s]: ##小于，往左子节点移动
            nearer_node = kd_node.left
            further_node = kd_node.right
        else:                         ##大于，往右子节点移动
            nearer_node = kd_node.right
            further_node = kd_node.left
        
        temp1 = travel(nearer_node, target_elt, max_dist)
        nearest = temp1.nearest_point
        dist = temp1.nearest_dist
        visited_nodes += temp1.visited_nodes
        
        if dist < max_dist:
            max_dist = dist
        
        temp_dist = abs(target_elt[s] - pivot[s])
        if temp_dist > max_dist:
            return result(nearest, dist, visited_nodes)
        
        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target_elt)))
        if temp_dist < max_dist:
            nearest = pivot
            dist = temp_dist
            max_dist = dist

        temp2 = travel(further_node, target_elt, max_dist)
        visited_nodes += temp2.visited_nodes
        
        if temp2.nearest_dist < dist:
            nearest = temp2.nearest_point
            dist = temp2.nearest_dist
        
        return result(nearest, dist, visited_nodes)
    return travel(tree.root, point_elt, float("inf"))
    
def bbf_find_nearest(tree, point_elt):
    pass


if __name__ == "__main__":
    data = [[2,3] , [5,4], [9,6], [4,7], [8,1], [7,2]]
    kdtree = KdTree(data)
    preOrder(kdtree.root)
    ret = find_nearest(kdtree, [3, 4.5])
    print(ret)

    