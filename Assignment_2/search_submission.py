# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

from __future__ import division

import heapq
import os
import pickle
import math


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
        current (int): The index of the current node in the queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        self.current = 0
        self.node_finder = {}
        self.REMOVED = '<removed-node>'
        self.just_sort = True

    def just_sort_pq(self, val):
        if val == True:
            self.just_sort = True
        else:
            self.just_sort = False
    
    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!
        if self.just_sort == True:
            weight, count, node = heapq.heappop(self.queue)
            return (weight, node)
        else:
            while self.queue:
                weight, count, node = heapq.heappop(self.queue)
                if (not(node == self.REMOVED)):
                    del self.node_finder[str(node)]
                    return (weight, count, node)
            raise KeyError('Pop from an empty priority queue')
    
    def is_node_in(self, node):
        return (self.node_finder.has_key(node)) and (not(node == self.REMOVED))
    
    def get_weight(self, node):
        node_entry = self.node_finder[node]
        return node_entry[0], node_entry[1]
    
    def compare(self, other):
        pass
    
    def get_last(self):
        return self.queue[-1]
        
    def copy(self):
        return self.queue
    
    def _return(self, index):
        return self.queue[index]

    def remove(self, node_id):
        """
        Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node_id (int): Index of node in queue.
        """
        node_entry = self.queue[node_id]
        node_to_be_removed = node_entry[-1]
        node_entry[-1] = self.REMOVED
        pop_it_out_of_list = self.node_finder.pop(str(node_to_be_removed))
        
    def is_empty(self):
        if not self.queue:
            return True
        else:
            return False
        
    def remove_node(self, node):
        
        removed_node_entry = self.node_finder.pop(str(node))
        removed_node_entry[-1] = self.REMOVED
                          
    def index(self, node):
        return self.queue.index(node)
    
    def _print_(self):
        print self.queue 
    

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        if self.node_finder.has_key(str(node[1])) :
            self.remove_node(node[1])
        count = self.current
        new_node_entry = [node[0], count, node[1]]
        self.node_finder[str(node[1])] = new_node_entry
        heapq.heappush(self.queue, new_node_entry)
        self.current += 1
        

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n for _, n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self == other

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]




def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    path = []
    frontier =[]
    node = start
    if node == goal:
        return path
    path.append([start])
    frontier.append(node)
    explored = []
    while True:
        temp_path = []
        if not frontier:
            return [None]
        node = frontier[0]
        frontier.remove(frontier[0])
        explored.append(node)
        temp_path = path[0]
        path.remove(path[0])
        for child in graph[node]:
            if not((child in frontier) or (child in explored)):
                temp_temp_path = temp_path[:]
                temp_temp_path.append(child)
                path.append(temp_temp_path)
                if child == goal:
                    return path[-1]
                frontier.append(child)
                
def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
#    print 'a'
#    print graph['a']
#    print "                              "
#    print 'z'
#    print graph['z']
#    print "                               "
#    print 's'
#    print graph['s']
#    print "                               "
#    print 't'
#    print graph['t']
#    print "                               "
#    print 'l'
#    print graph['l']
#    print "                                "
#    print 'm'
#    print graph['m']
#    print "                                "
#    print 'r'
#    print graph['r']
#    print "                                 "
#    print 'c'
#    print graph['c']
#    print "                                 "
#    print 'd'
#    print graph['d']
#    print "                                 "
#    print 'f'
#    print graph['f']
#    print "                                 "
#    print 'p'
#    print graph['p']
#    print "                                 "
#    print 'b'
#    print graph['b']
#    print "                                 "
#    print 'o'
#    print graph['o']
#    print "                                  "
    path = PriorityQueue()
    frontier = PriorityQueue()
    frontier.just_sort_pq(False)
    path.just_sort_pq(False)
    node = start
    frontier.append((0,start))
    path.append((0,str([])))
    explored = []
    count = 0
    while True:
        temp_path=[]
        if frontier.is_empty():
            return [None]
        weight, count, node = frontier.pop()
        if node == goal:
            priority,path_count, path_to_return = path.pop()
            path_to_return = eval(path_to_return)
            return path_to_return
        explored.append(node)
        if count == 0:
            interim_weight, interim_count, interim_path = path.pop()
            interim_path = eval(interim_path)
            interim_path.append(node)
            path.append((interim_weight,str(interim_path)))
            count += 1
        temp_weight,temp_count, temp_path = path.pop()
        temp_path = eval(temp_path)
        for child, weight_dict in graph[node].items():
            child_weight = weight_dict['weight']
            if not((child in explored) or (frontier.is_node_in(child))):
                frontier.append((temp_weight+child_weight, child))
                temp_temp_path = temp_path[:]
                temp_temp_path.append(child)
                path.append((temp_weight+child_weight,str(temp_temp_path)))
            elif frontier.is_node_in(child) and (frontier.get_weight(child)[0] > (temp_weight+child_weight)):
                actual_weight, actual_count = frontier.get_weight(child)
                index_in_frontier = frontier.index([actual_weight, actual_count, child])
                path.remove(index_in_frontier)
                frontier.remove_node(child)
                frontier.append((temp_weight+child_weight, child))
                temp_temp_path = temp_path[:]
                temp_temp_path.append(child)
                path.append((temp_weight+child_weight,str(temp_temp_path)))
                

def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    pos_v = graph.node[v]['pos']
    pos_goal = graph.node[goal]['pos']
    return math.sqrt(math.pow((pos_v[0]-pos_goal[0]), 2) + math.pow((pos_v[1]-pos_goal[1]), 2))


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    frontier = PriorityQueue()
    frontier.just_sort_pq(False)
    path = PriorityQueue()
    path.just_sort_pq(False)    
    node = start
    frontier.append(((heuristic(graph, node, goal), 0), node))
    path.append(((heuristic(graph, node, goal), 0), []))
    explored = []
    count = 0
    while True:
        if frontier.is_empty():
            return [None]
        weight, count, node = frontier.pop()
        if node == goal:
            priority, path_count, path_to_return = path.pop()
#            path_to_return = eval(path_to_return)
            return path_to_return
        explored.append(node)
        if count == 0:
            interim_weight, interim_count, interim_path = path.pop()
#            interim_path = eval(interim_path)
            interim_path.append(node)
#            path.append((interim_weight, str(interim_path)))
            path.append((interim_weight, interim_path))
            count += 1
        temp_weight,temp_count, temp_path = path.pop()
#        temp_path = eval(temp_path)
        for child, weight_dict in graph[node].items():
            child_weight = weight_dict['weight']
            child_h = heuristic(graph, child, goal)
            child_g = temp_weight[1]+child_weight
            if not((child in explored) or (frontier.is_node_in(child))):
                frontier.append(((child_h+child_g, child_g), child))
                temp_temp_path = temp_path[:]
                temp_temp_path.append(child)
#                path.append(((child_h+child_g, child_g), str(temp_temp_path)))
                path.append(((child_h+child_g, child_g), temp_temp_path))
#            elif frontier.is_node_in(child) and (frontier.get_weight(child)[0][0] > (temp_weight[1]+child_weight)):
            elif frontier.is_node_in(child) and (frontier.get_weight(child)[0][0] > (child_h + child_g)):
                actual_weight, actual_count = frontier.get_weight(child)
                index_in_frontier = frontier.index([actual_weight, actual_count, child])
                path.remove(index_in_frontier)
                frontier.remove_node(child)
                frontier.append(((child_h+child_g, child_g), child))
                temp_temp_path = temp_path[:]
                temp_temp_path.append(child)
#                path.append(((child_h+child_g, child_g), str(temp_temp_path)))
                path.append(((child_h+child_g, child_g), temp_temp_path))

def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    path_F = PriorityQueue()
    path_R = PriorityQueue()
    frontier_F = PriorityQueue()
    frontier_R = PriorityQueue()
    frontier_F.just_sort_pq(False)
    frontier_R.just_sort_pq(False)
    path_F.just_sort_pq(False)
    path_R.just_sort_pq(False)
    node_F = start
    node_R = goal
    frontier_F.append((0,start))
    frontier_R.append((0,goal))
    path_F.append(((0, node_F),[]))
    path_R.append(((0, node_R),[]))
    path_to_process_F = []
    path_to_process_R = []
    explored_F = []
    explored_R = []
    explored_dict_F = {}
    explored_dict_R = {}
    count_s = 0
    count_g = 0
    top_R = 0
    mu = float("inf")
    path = []
    while True:
        if frontier_F.is_empty() and frontier_R.is_empty():
            return [None]
        if not(frontier_F.is_empty()):
            cost_F, count_F, node_F = frontier_F.pop()
            top_F = cost_F
            print "top_F + top_R", top_F + top_R 
            print "Mu:  ", mu
            if(node_F == goal) or ((top_F + top_R) > mu) or (node_R == 'end'):
                Entire_path_F = path_F.copy()
                Entire_path_R = path_R.copy()
                if path_to_process_F:
                    for entry_F in path_to_process_F:
                        cost_potential_F = entry_F[0][0]
                        if path_to_process_R:
                            for entry_R in path_to_process_R:
                                cost_potential_R = entry_R[0][0]
                                if(cost_potential_F + cost_potential_R) == mu:
                                    path = entry_F[-1][:-1] + entry_R[-1][::-1]
                                    return path
                        for potential_path_entry_R in Entire_path_R:
                            potential_path_R = potential_path_entry_R[-1]
                            if not(potential_path_R == '<removed-node>'):
                                cost_potential_R = potential_path_entry_R[0][0]
                                if(cost_potential_F+cost_potential_R) == mu:
                                    path = entry_F[-1][:-1] + potential_path_R[::-1]
                                    return path
                if path_to_process_R:
                    for entry_R in path_to_process_R:
                        cost_potential_R = entry_R[0][0]
                        for potential_path_entry_F in Entire_path_F:
                            potential_path_F = potential_path_entry_F[-1]
                            if not(potential_path_F == '<removed-node>'):
                                cost_potential_F = potential_path_entry_F[0][0]
                                if(cost_potential_F + cost_potential_R) == mu:
                                    path = potential_path_F[:-1] + entry_R[-1][::-1]
                                    return path
                for potential_path_entry_F in Entire_path_F:
                    potential_path_F = potential_path_entry_F[-1]
                    if not(potential_path_F == '<removed-node>'):
                        cost_potential_F = potential_path_entry_F[0][0]
                        for potential_path_entry_R in Entire_path_R:
                            potential_path_R = potential_path_entry_R[-1]
                            if not(potential_path_R == '<removed-node>'):
                                cost_potential_R = potential_path_entry_R[0][0]
                                if(cost_potential_F+cost_potential_R) == mu:
                                    path = potential_path_F[:-1] + potential_path_R
                                    return path
                print path
                return path
            explored_F.append(node_F)
            explored_dict_F[node_F] = cost_F
    
            if count_s == 0:
                i_cost_F, i_count_F, i_path_F = path_F.pop()
                i_path_F.append(node_F)
                path_F.append((i_cost_F, i_path_F))
                count_s += 1
            parent_cost_F, parent_count_F, parent_path_F = path_F.pop()
            for child_F in graph[node_F]:
                child_F_step_cost = graph[node_F][child_F]['weight']
                child_g_F = parent_cost_F[0] + child_F_step_cost
                if not(explored_dict_F.has_key(child_F) or frontier_F.is_node_in(child_F)):
                    if explored_dict_R.has_key(child_F):
                        path_to_process_F.append([parent_cost_F, parent_path_F])
                        print path_to_process_F
    #                    frontier_F.append((float("inf"), 'end'))
    #                    path_F.append((parent_cost_F, parent_path_F))
                        if (child_g_F + explored_dict_R[child_F]) < mu:
                            mu = child_g_F + explored_dict_R[child_F]
                    else:
                        frontier_F.append((child_g_F, child_F))
                        temp_path_F = parent_path_F[:]
                        temp_path_F.append(child_F)
                        path_F.append(((child_g_F, child_F), temp_path_F))
                elif frontier_F.is_node_in(child_F) and (frontier_F.get_weight(child_F)[0] > child_g_F):
                    actual_cost_F, actual_count_F = frontier_F.get_weight(child_F)
                    indx_in_F = frontier_F.index([actual_cost_F, actual_count_F, child_F])
                    path_F.remove(indx_in_F)
                    frontier_F.remove_node(child_F)
                    frontier_F.append((child_g_F, child_F))
                    temp_path_F = parent_path_F[:]
                    temp_path_F.append(child_F)
                    path_F.append(((child_g_F, child_F), temp_path_F))
                    if explored_dict_R.has_key(child_F):
                        if (child_g_F + explored_dict_R[child_F]) < mu:
                            mu = child_g_F + explored_dict_R[child_F]
                
##################################################################################################################################
##################################################################################################################################
        if not(frontier_R.is_empty()):       
            cost_R, count_R, node_R = frontier_R.pop()
            top_R = cost_R
            print "top_F + top_R", top_F + top_R 
            print "Mu:  ", mu
            if(node_R == start) or ((top_F + top_R) > mu) or (node_R == 'end'):
                Entire_path_F = path_F.copy()
                Entire_path_R = path_R.copy()
                if path_to_process_F:
                    for entry_F in path_to_process_F:
                        cost_potential_F = entry_F[0][0]
                        if path_to_process_R:
                            for entry_R in path_to_process_R:
                                cost_potential_R = entry_R[0][0]
                                if(cost_potential_F + cost_potential_R) == mu:
                                    path = entry_F[-1][:-1] + entry_R[-1][::-1]
                                    return path
                        for potential_path_entry_R in Entire_path_R:
                            potential_path_R = potential_path_entry_R[-1]
                            if not(potential_path_R == '<removed-node>'):
                                cost_potential_R = potential_path_entry_R[0][0]
                                if(cost_potential_F+cost_potential_R) == mu:
                                    path = entry_F[-1][:-1] + potential_path_R[::-1]
                                    return path
                if path_to_process_R:
                    for entry_R in path_to_process_R:
                        cost_potential_R = entry_R[0][0]
                        for potential_path_entry_F in Entire_path_F:
                            potential_path_F = potential_path_entry_F[-1]
                            if not(potential_path_F == '<removed-node>'):
                                cost_potential_F = potential_path_entry_F[0][0]
                                if(cost_potential_F + cost_potential_R) == mu:
                                    path = potential_path_F[:-1] + entry_R[-1][::-1]
                                    return path
                for potential_path_entry_F in Entire_path_F:
                    potential_path_F = potential_path_entry_F[-1]
                    if not(potential_path_F == '<removed-node>'):
                        cost_potential_F = potential_path_entry_F[0][0]
                        for potential_path_entry_R in Entire_path_R:
                            potential_path_R = potential_path_entry_R[-1]
                            if not(potential_path_R == '<removed-node>'):
                                cost_potential_R = potential_path_entry_R[0][0]
                                if(cost_potential_F+cost_potential_R) == mu:
                                    path = potential_path_F[:-1] + potential_path_R
                                    return path
    
                print path
                return path
            explored_R.append(node_R)
            explored_dict_R[node_R] = cost_R
    
            if count_g == 0:
                i_cost_R, i_count_R, i_path_R = path_R.pop()
                i_path_R.append(node_R)
                path_R.append((i_cost_R, i_path_R))
                count_g += 1
            parent_cost_R, parent_count_R, parent_path_R = path_R.pop()
            for child_R in graph[node_R]:
                child_R_step_cost = graph[node_R][child_R]['weight']
                child_g_R = parent_cost_R[0] + child_R_step_cost
                if not(explored_dict_R.has_key(child_R) or frontier_R.is_node_in(child_R)):
                    if explored_dict_F.has_key(child_R):
                        path_to_process_R.append([parent_cost_R, parent_path_R])
                        print path_to_process_R
    #                    frontier_R.append((float("inf"), 'end'))
    #                    path_R.append((parent_cost_R, parent_path_R))
                        if (child_g_R + explored_dict_F[child_R]) < mu:
                            mu = child_g_R + explored_dict_F[child_R]
                    else:
                        frontier_R.append((child_g_R, child_R))
                        temp_path_R = parent_path_R[:]
                        temp_path_R.append(child_R)
                        path_R.append(((child_g_R, child_R), temp_path_R))
                elif frontier_R.is_node_in(child_R) and (frontier_R.get_weight(child_R)[0] > child_g_R):
                    actual_cost_R, actual_count_R = frontier_R.get_weight(child_R)
                    indx_in_R = frontier_R.index([actual_cost_R, actual_count_R, child_R])
                    path_R.remove(indx_in_R)
                    frontier_R.remove_node(child_R)
                    frontier_R.append((child_g_R, child_R))
                    temp_path_R = parent_path_R[:]
                    temp_path_R.append(child_R)
                    path_R.append(((child_g_R, child_R), temp_path_R))
                    if explored_dict_F.has_key(child_R):
                        if (child_g_R + explored_dict_F[child_R]) < mu:
                            mu = child_g_R + explored_dict_F[child_R]
                    
               

#def bidirectional_a_star(graph, start, goal,
#                         heuristic=euclidean_dist_heuristic):
#    """
#    Exercise 2: Bidirectional A*.
#
#    See README.md for exercise description.
#
#    Args:
#        graph (ExplorableGraph): Undirected graph to search.
#        start (str): Key for the start node.
#        goal (str): Key for the end node.
#        heuristic: Function to determine distance heuristic.
#            Default: euclidean_dist_heuristic.
#
#    Returns:
#        The best path as a list from the start and goal nodes (including both).
#    """
#
#    # TODO: finish this function!
#    def pf(v):
#        return 0.5*(heuristic(graph,v,goal)-heuristic(graph,v,start)) + 0.5*heuristic(graph,goal,start)
# #       return 0.5*(heuristic(graph,v,goal)-heuristic(graph,v,start))
#    
#    def pr(v):
#        return 0.5*(heuristic(graph,v,start)-heuristic(graph,v,goal)) + 0.5*heuristic(graph,start,goal)
##        return 0.5*(heuristic(graph,v,start)-heuristic(graph,v,goal))
#    
#    mu = float("inf")
#    frontier_F = PriorityQueue()
#    frontier_F.just_sort_pq(False)
#    frontier_R = PriorityQueue()
#    frontier_R.just_sort_pq(False)
#    path_F = PriorityQueue()
#    path_F.just_sort_pq(False)
#    path_R = PriorityQueue()
#    path_R.just_sort_pq(False)
#    node_F = start
#    node_R = goal
#    frontier_F.append(((heuristic(graph,node_F,goal), 0),node_F))
#    frontier_R.append(((heuristic(graph,node_R,start),0),node_R))
#    path_F.append(((heuristic(graph,node_F,goal),0),str([])))
#    path_R.append(((heuristic(graph,node_R,start),0),str([])))
#    explored_dict_F = {}
#    explored_dict_R = {}
#    explored_F = []
#    explored_R = []
#    count_F = 0
#    count_R = 0
#    path = [None]
#    top_F = 0
#    top_R = 0
#    
#    while True:
#        if frontier_F.is_empty() or frontier_R.is_empty():
#            return [None]
#        cost_F, counter_F, node_F = frontier_F.pop()
#        top_F = cost_F[1] + pf(node_F)
#        if (node_F == goal) or ((top_F + top_R) >= (mu + pr(goal))):
#            explored = explored_F + explored_R
#            possible_path_F = path_F.copy()
#            possible_path_R = path_R.copy()
#            cost_check = float("inf")
#            for potential_path_entry_F in possible_path_F:
#                potential_path_F = potential_path_entry_F[-1]
#                if not(potential_path_F == "<removed-node>"):
#                    cost_potential_path_F = potential_path_entry_F[0][1]
#                    potential_path_F = eval(potential_path_F)
#                    for potential_path_entry_R in possible_path_R:
#                        potential_path_F = potential_path_entry_F[-1]
#                        potential_path_F = eval(potential_path_F)
#                        potential_path_R = potential_path_entry_R[-1]
#                        if not(potential_path_R == "<removed-node>"):
#                            cost_potential_path_R = potential_path_entry_R[0][1]
#                            potential_path_R = eval(potential_path_R)
#                            potential_path_R_loop = potential_path_R[::-1]
#
#                            path_intersect = list(set(potential_path_F).intersection(potential_path_R_loop))
#                            if(len(path_intersect) >= 1):
#                                cost_to_remove = 0
#                                if(len(path_intersect) >= 2):
#                                    to_remove = potential_path_F[-len(path_intersect):]
#                                    for node1, node2 in zip(to_remove, to_remove[1:]):
#                                        cost_to_remove += graph[node1][node2]['weight']
#                                cost_check_compare = cost_potential_path_F + cost_potential_path_R - cost_to_remove
#                                if cost_check_compare < cost_check:
#
#                                    potential_path_R_loop = potential_path_R_loop[len(path_intersect):]
#                                    if not(potential_path_F[-1] in path_intersect):
#                                        potential_path_F = potential_path_F[:-len(path_intersect)]
#                                    if(len(potential_path_R_loop)>0) and (not(potential_path_F[-1] == goal)):
#                                        if(potential_path_F[-1] in graph.neighbors(potential_path_R_loop[0])) and (potential_path_F[-1] in explored)\
#                                                                                          and (potential_path_R_loop[0] in explored):
#
#                                            if (potential_path_R_loop[0] == start) and (potential_path_R_loop[-1] == goal):
#                                                path = potential_path_R_loop
#                                            else:
#                                                path = potential_path_F + potential_path_R_loop
#                                                
#                                        cost_check = cost_check_compare
#                                    else:
#                                        path = potential_path_F
#                                        cost_check = cost_check_compare
#            return path
#        explored_F.append(node_F)
#        if count_F == 0:
#            i_cost_F, i_counter_F, i_path_F = path_F.pop()
#            i_path_F = eval(i_path_F)
#            i_path_F.append(node_F)
#            path_F.append((i_cost_F, str(i_path_F)))
#            count_F +=1
#        parent_cost_F, parent_count_F, parent_path_F = path_F.pop()
#        parent_path_F = eval(parent_path_F)
#        for child_F, weight_dict_F in graph[node_F].items():
#            child_step_cost_F = weight_dict_F['weight']
#            child_h_F = heuristic(graph, child_F, goal)
#            child_g_F = parent_cost_F[1] + child_step_cost_F
#            if not((child_F in explored_F) or (frontier_F.is_node_in(child_F))):
#                frontier_F.append(((child_h_F+child_g_F, child_g_F), child_F))
#                temp_path_F = parent_path_F[:]
#                temp_path_F.append(child_F)
#                path_F.append(((child_h_F+child_g_F, child_g_F), str(temp_path_F)))
#            elif frontier_F.is_node_in(child_F) and (frontier_F.get_weight(child_F)[0][0] > (child_h_F + child_g_F)):
#                actual_cost_F, actual_count_F = frontier_F.get_weight(child_F)
#                idx_in_F = frontier_F.index([actual_cost_F, actual_count_F, child_F])
#                path_F.remove(idx_in_F)
#                frontier_F.remove_node(child_F)
#                frontier_F.append(((child_h_F+child_g_F, child_g_F), child_F))
#                temp_path = parent_path_F[:]
#                temp_path.append(child_F)
#                path_F.append(((child_h_F+child_g_F, child_g_F), str(temp_path)))
#            if not(child_F in explored_F):
#                explored_dict_F[(node_F, child_F)] = parent_cost_F[1]
#                if explored_dict_R.has_key((child_F, node_F)):
#                    if (child_g_F + explored_dict_R[(child_F, node_F)]) < mu:
#                        mu = child_g_F + explored_dict_R[(child_F, node_F)]
########################################################################################################################################        
#        cost_R, counter_R, node_R = frontier_R.pop()
#        top_R = cost_R[1] + pr(node_R)
#        if (node_R == start) or ((top_F + top_R) >= (mu + pr(goal))):
#            explored = explored_F + explored_R            
#            possible_path_F = path_F.copy()
#            possible_path_R = path_R.copy()
#            cost_check = float("inf")
#            for potential_path_entry_F in possible_path_F:
#                potential_path_F = potential_path_entry_F[-1]
#                if not(potential_path_F == "<removed-node>"):
#                    cost_potential_path_F = potential_path_entry_F[0][1]
#                    potential_path_F = eval(potential_path_F)
#                    for potential_path_entry_R in possible_path_R:
#                        potential_path_F = potential_path_entry_F[-1]
#                        potential_path_F = eval(potential_path_F)
#                        potential_path_R = potential_path_entry_R[-1]
#                        if not(potential_path_R == "<removed-node>"):
#                            cost_potential_path_R = potential_path_entry_R[0][1]
#                            potential_path_R = eval(potential_path_R)
#                            potential_path_R = potential_path_R[::-1]
#                            path_intersect = list(set(potential_path_F).intersection(potential_path_R))
#                            if(len(path_intersect) >= 1):
#                                cost_to_remove = 0
#                                if(len(path_intersect) >= 2):
#                                    to_remove = potential_path_F[-len(path_intersect):]
#                                    for node1, node2 in zip(to_remove, to_remove[1:]):
#                                        cost_to_remove += graph[node1][node2]['weight']
#                                cost_check_compare = cost_potential_path_F + cost_potential_path_R - cost_to_remove
#                                if cost_check_compare < cost_check:
#                                    potential_path_R = potential_path_R[len(path_intersect):]
#                                    if not(potential_path_F[-1] in path_intersect):
#                                        potential_path_F = potential_path_F[:-len(path_intersect)]
#                                    if(len(potential_path_R)>0) and (not(potential_path_F[-1] == goal)):
#                                        if(potential_path_F[-1] in graph.neighbors(potential_path_R[0])) and (potential_path_F[-1] in explored)\
#                                                                                          and (potential_path_R[0] in explored):
#
#                                            if (potential_path_R[0] == start) and (potential_path_R[-1] == goal):
#                                                path = potential_path_R
#                                            else:
#                                                path = potential_path_F + potential_path_R
#                                                
#                                        cost_check = cost_check_compare
#                                    else:
#                                        path = potential_path_F
#                                        cost_check = cost_check_compare
#            return path
#
#        explored_R.append(node_R)
#        if count_R == 0:
#            i_cost_R, i_counter_R, i_path_R = path_R.pop()
#            i_path_R = eval(i_path_R)
#            i_path_R.append(node_R)
#            path_R.append((i_cost_R, str(i_path_R)))
#            count_R +=1
#        parent_cost_R, parent_count_R, parent_path_R = path_R.pop()
#        parent_path_R = eval(parent_path_R)
#        for child_R, weight_dict_R in graph[node_R].items():
#            child_step_cost_R = weight_dict_R['weight']
#            child_h_R = heuristic(graph, child_R, start)
#            child_g_R = parent_cost_R[1] + child_step_cost_R
#            if not((child_R in explored_R) or frontier_R.is_node_in(child_R)):
#
#                frontier_R.append(((child_h_R+child_g_R, child_g_R), child_R))
#                temp_path_R = parent_path_R[:]
#                temp_path_R.append(child_R)
#                path_R.append(((child_h_R+child_g_R, child_g_R), str(temp_path_R)))
#            elif frontier_R.is_node_in(child_R) and (frontier_R.get_weight(child_R)[0][0] > (child_h_R + child_g_R)):
#                actual_cost_R, actual_count_R = frontier_R.get_weight(child_R)
#                idx_in_R = frontier_R.index([actual_cost_R, actual_count_R, child_R])
#                path_R.remove(idx_in_R)
#                frontier_R.remove_node(child_R)
#                frontier_R.append(((child_h_R+child_g_R, child_g_R), child_R))
#                temp_path_R = parent_path_R[:]
#                temp_path_R.append(child_R)
#                path_R.append(((child_h_R+child_g_R, child_g_R), str(temp_path_R)))
#            if not(child_R in explored_R):
#                explored_dict_R[(node_R, child_R)] = parent_cost_R[1]
#                if explored_dict_F.has_key((child_R, node_R)):
#                    if (child_g_R + explored_dict_F[(child_R, node_R)]) < mu:
#                        mu = child_g_R + explored_dict_F[(child_R, node_R)]
            
#*******************************************************************************
#********************************************************************************
#def bidirectional_a_star(graph, start, goal,
#                         heuristic=euclidean_dist_heuristic):
def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    frontier_F = PriorityQueue()
    frontier_F.just_sort_pq(False)
    frontier_R = PriorityQueue()
    frontier_R.just_sort_pq(False)
    path_F = PriorityQueue()
    path_F.just_sort_pq(False)
    path_R = PriorityQueue()
    path_R.just_sort_pq(False)
    node_F = start
    node_R = goal
    frontier_F.append(((heuristic(graph,node_F,goal), 0),node_F))
    frontier_R.append(((heuristic(graph,node_R,start),0),node_R))
    path_F.append(((heuristic(graph,node_F,goal),0, node_F),[]))
    path_R.append(((heuristic(graph,node_R,start),0, node_R),[]))
    explored_dict_F = {}
    explored_dict_R = {}
    explored_F = []
    explored_R = []
    path_to_process_F = []
    path_to_process_R = []
    count_F = 0
    count_R = 0
    explored_intersect = 0
    path = []


    while True:

        if frontier_F.is_empty() or frontier_R.is_empty():
            return [None]
        cost_F, counter_F, node_F = frontier_F.pop()

        if (node_F == goal) or (explored_intersect >= 1) or (node_F == 'end'):
            cost_check = float("inf")
            entire_path_F = path_F.copy()
            entire_path_R = path_R.copy()
            if path_to_process_F:
                for entry_F in path_to_process_F:
                    potential_intersect_F = entry_F[0][2]
                    cost_potential_F = entry_F[0][1]
                    potential_path_F = entry_F[-1]
                    if (potential_path_F[0] == start) and (potential_path_F[-1] == goal) and (cost_potential_F < cost_check):
                        path = potential_path_F
                        cost_check = cost_potential_F
                    if path_to_process_R:
                        for entry_R in path_to_process_R:
                            if entry_R[0][2] == potential_intersect_F:
                                cost_potential_R = entry_R[0][1]
                                if (cost_potential_F + cost_potential_R) < cost_check:
                                    potential_path_R = entry_R[-1]
                                    path = potential_path_F[:-1] + potential_path_R[::-1]
                                    cost_check = cost_potential_F + cost_potential_R
                    
                    for potential_path_entry_R in entire_path_R:
                        potential_path_R = potential_path_entry_R[-1]
                        if not(potential_path_R == "<removed-node>"):
#                            potential_path_R = eval(potential_path_R)
                            if potential_path_R:
                                if potential_path_R[-1] == 'end':
                                    potential_path_R = potential_path_R[:-1]
                                if potential_path_entry_R[0][2] == potential_intersect_F:
                                    cost_potential_R = potential_path_entry_R[0][1]
                                    if (cost_potential_F + cost_potential_R) < cost_check:
                                        path = potential_path_F[:-1] + potential_path_R[::-1]
                                        cost_check = cost_potential_F + cost_potential_R
            
            if path_to_process_R:
                for entry_R in path_to_process_R:
                    potential_intersect_R = entry_R[0][2]
                    cost_potential_R = entry_R[0][1]
                    potential_path_R = entry_R[-1]
                    if(potential_path_R[-1] == start) and (potential_path_R[0] == goal) and (cost_potential_R < cost_check):
                        path = potential_path_R[::-1]
                        cost_check = cost_potential_R
                    
                    for potential_path_entry_F in entire_path_F:
                        potential_path_F = potential_path_entry_F[-1]
                        if not(potential_path_F == "<removed-node>"):
                            if potential_path_F:
                                if potential_path_F[-1] == 'end':
                                    potential_path_F = potential_path_F[:-1]
                                if potential_path_entry_F[0][2] == potential_intersect_R:
                                    cost_potential_F = potential_path_entry_F[0][1]
                                    if (cost_potential_F+cost_potential_R) < cost_check:
                                        path = potential_path_F[:-1] + potential_path_R[::-1]
                                        cost_check = cost_potential_F + cost_potential_R
            return path

        explored_F.append(node_F)
        explored_dict_F[node_F] = True
        if explored_dict_R.has_key(node_F):
            explored_intersect += 1
        if count_F == 0:
            i_cost_F, i_counter_F, i_path_F = path_F.pop()
            i_path_F.append(node_F)
            path_F.append((i_cost_F, i_path_F))
            count_F +=1
        parent_cost_F, parent_count_F, parent_path_F = path_F.pop()
        for child_F in graph[node_F]:
            child_step_cost_F = graph[node_F][child_F]['weight']
            child_h_F = heuristic(graph, child_F, goal)
            child_g_F = parent_cost_F[1] + child_step_cost_F
            if not((explored_dict_F.has_key(child_F)) or (frontier_F.is_node_in(child_F))):
                if explored_dict_R.has_key(child_F):
                    path_to_process_F.append([parent_cost_F, parent_path_F])
                    frontier_F.append((((float("inf"), parent_cost_F[1]), 'end')))
                    path_F.append(((float("inf"), parent_cost_F[1], node_F), parent_path_F))

                else:
                    frontier_F.append(((child_h_F+child_g_F, child_g_F), child_F))
                    temp_path_F = parent_path_F[:]
                    temp_path_F.append(child_F)
                    path_F.append(((child_h_F+child_g_F, child_g_F, child_F), temp_path_F))
            elif frontier_F.is_node_in(child_F) and (frontier_F.get_weight(child_F)[0][0] > (child_h_F + child_g_F)):
                actual_cost_F, actual_count_F = frontier_F.get_weight(child_F)
                idx_in_F = frontier_F.index([actual_cost_F, actual_count_F, child_F])
                path_F.remove(idx_in_F)
                frontier_F.remove_node(child_F)
                frontier_F.append(((child_h_F+child_g_F, child_g_F), child_F))
                temp_path = parent_path_F[:]
                temp_path.append(child_F)
                path_F.append(((child_h_F+child_g_F, child_g_F, child_F), temp_path))

        cost_R, counter_R, node_R = frontier_R.pop()
        if (node_R == start) or (explored_intersect >= 1) or (node_R == 'end'):
            cost_check = float("inf")
            entire_path_F = path_F.copy()
            entire_path_R = path_R.copy()
            if path_to_process_F:
                for entry_F in path_to_process_F:
                    potential_intersect_F = entry_F[0][2]
                    cost_potential_F = entry_F[0][1]
                    potential_path_F = entry_F[-1]
                    if (potential_path_F[0] == start) and (potential_path_F[-1] == goal) and (cost_potential_F < cost_check):
                        path = potential_path_F
                        cost_check = cost_potential_F
                    if path_to_process_R:
                        for entry_R in path_to_process_R:
                            if entry_R[0][2] == potential_intersect_F:
                                cost_potential_R = entry_R[0][1]
                                if (cost_potential_F + cost_potential_R) < cost_check:
                                    potential_path_R = entry_R[-1]
                                    path = potential_path_F[:-1] + potential_path_R[::-1]
                                    cost_check = cost_potential_F + cost_potential_R
                    
                    for potential_path_entry_R in entire_path_R:
                        potential_path_R = potential_path_entry_R[-1]
                        if not(potential_path_R == "<removed-node>"):
                            if potential_path_R:
                                if potential_path_R[-1] == 'end':
                                    potential_path_R = potential_path_R[:-1]
                                if potential_path_entry_R[0][2] == potential_intersect_F:
                                    cost_potential_R = potential_path_entry_R[0][1]
                                    if (cost_potential_F + cost_potential_R) < cost_check:
                                        path = potential_path_F[:-1] + potential_path_R[::-1]
                                        cost_check = cost_potential_F + cost_potential_R
            
            if path_to_process_R:
                for entry_R in path_to_process_R:
                    potential_intersect_R = entry_R[0][2]
                    cost_potential_R = entry_R[0][1]
                    potential_path_R = entry_R[-1]
                    if(potential_path_R[-1] == start) and (potential_path_R[0] == goal) and (cost_potential_R < cost_check):
                        path = potential_path_R[::-1]
                        cost_check = cost_potential_R
                    
                    for potential_path_entry_F in entire_path_F:
                        potential_path_F = potential_path_entry_F[-1]
                        if not(potential_path_F == "<removed-node>"):
                            if potential_path_F:
                                if potential_path_F[-1] == 'end':
                                    potential_path_F = potential_path_F[:-1]
                                if potential_path_entry_F[0][2] == potential_intersect_R:
                                    cost_potential_F = potential_path_entry_F[0][1]
                                    if (cost_potential_F+cost_potential_R) < cost_check:
                                        path = potential_path_F[:-1] + potential_path_R[::-1]
                                        cost_check = cost_potential_F + cost_potential_R
            return path
        explored_R.append(node_R)
        explored_dict_R[node_R] = True
        if explored_dict_F.has_key(node_R):
            explored_intersect += 1
        if count_R == 0:
            i_cost_R, i_counter_R, i_path_R = path_R.pop()
            i_path_R.append(node_R)
            path_R.append((i_cost_R, i_path_R))
            count_R +=1
        parent_cost_R, parent_count_R, parent_path_R = path_R.pop()
        for child_R in graph[node_R]:
            child_step_cost_R = graph[node_R][child_R]['weight']
            child_h_R = heuristic(graph, child_R, start)
            child_g_R = parent_cost_R[1] + child_step_cost_R
            if not((explored_dict_R.has_key(child_R)) or frontier_R.is_node_in(child_R)):
                if explored_dict_F.has_key(child_R):
                    path_to_process_R.append([parent_cost_R, parent_path_R])
                    frontier_R.append((((float("inf"), parent_cost_R[1]), 'end')))
                    path_R.append(((float("inf"), parent_cost_R[1], node_R), parent_path_R))
                else:
                    frontier_R.append(((child_h_R+child_g_R, child_g_R), child_R))
                    temp_path_R = parent_path_R[:]
                    temp_path_R.append(child_R)
                    path_R.append(((child_h_R+child_g_R, child_g_R, child_R), temp_path_R))
            elif frontier_R.is_node_in(child_R) and (frontier_R.get_weight(child_R)[0][0] > (child_h_R + child_g_R)):
                actual_cost_R, actual_count_R = frontier_R.get_weight(child_R)
                idx_in_R = frontier_R.index([actual_cost_R, actual_count_R, child_R])
                path_R.remove(idx_in_R)
                frontier_R.remove_node(child_R)
                frontier_R.append(((child_h_R+child_g_R, child_g_R), child_R))
                temp_path_R = parent_path_R[:]
                temp_path_R.append(child_R)
                path_R.append(((child_h_R+child_g_R, child_g_R, child_R), temp_path_R))

def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    path_A = PriorityQueue()
    frontier_A = PriorityQueue()
    path_A.just_sort_pq(False)
    frontier_A.just_sort_pq(False)
    path_B = PriorityQueue()
    frontier_B = PriorityQueue()
    path_B.just_sort_pq(False)
    frontier_B.just_sort_pq(False)
    path_C = PriorityQueue()
    frontier_C = PriorityQueue()
    path_C.just_sort_pq(False)
    frontier_C.just_sort_pq(False)
    node_A = goals[0]
    node_B = goals[1]
    node_C = goals[2]
    frontier_A.append((0,node_A))
    frontier_B.append((0,node_B))
    frontier_C.append((0,node_C))
    path_A.append(((0,node_A),[]))
    path_B.append(((0,node_B),[]))
    path_C.append(((0,node_C),[]))
    path_to_process_AB = []
    path_to_process_AC = []
    path_to_process_BC = []
    path_to_process_BA = []
    path_to_process_CA = []
    path_to_process_CB = []
    
    
    explored_dict_A = {}
    explored_dict_B = {}
    explored_dict_C = {}
    count_As = 0
    count_Bs = 0
    count_Cs = 0
    mu_AB = float("inf")
    mu_AC = float("inf")
    mu_BC = float("inf")
    top_A = 0
    top_B = 0
    top_C = 0
    path = []
    AB_done = False
    AC_done = False
    BC_done = False
    
    while True:
        if frontier_A.is_empty() and frontier_B.is_empty() and frontier_C.is_empty():
            return [None]
        
        if not(frontier_A.is_empty()):
            cost_A, count_A, node_A = frontier_A.pop()
            top_A = cost_A
            if (node_A == goals[1]) or ((top_A + top_B) > mu_AB):
                AB_done = True
                if(AB_done and AC_done) and (max(mu_AB, mu_AC) <= mu_BC):
                    print "******* A SEARCH ***************"
                    print "        "
                    print "path A"
                    path_A._print_()
                    print "    "
                    print "Path B"
                    path_B._print_()
                    print "    "
                    print "Path C"
                    path_C._print_()
                    print "    "
                    print "Path to process AB"
                    print path_to_process_AB
                    print "        "
                    print "Path to process BA"
                    print path_to_process_BA
                    print "    "
                    print "Path to process AC"
                    print path_to_process_AC
                    print "     "
                    print "Path to process CA"
                    print path_to_process_CA
                    print "    "
                    print "Path to process BC"
                    print path_to_process_BC
                    print "    "
                    print "Path to process CB"
                    print path_to_process_CB
                    print "         "
                    print "mu_AB:  ", mu_AB, "mu_AC:  ", mu_AC, "mu_BC:   ", mu_BC
                    print "**********************************"
                    Entire_path_A = path_A.copy()
                    Entire_path_B = path_B.copy()
                    Entire_path_C = path_C.copy()
                    if path_to_process_AB:
                        for entry_AB in path_to_process_AB:
                            cost_potential_AB = entry_AB[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if (cost_potential_AB+cost_potential_B) == mu_AB:
                                        path_AB = entry_AB[-1][:-1] + potential_path_B[::-1]
                                        break
                            if path_to_process_BA:
                                for entry_BA in path_to_process_BA:
                                    cost_potential_BA = entry_BA[0][0]
                                    if(cost_potential_AB+cost_potential_BA) == mu_AB:
                                        path_AB = entry_AB[-1][:-1] + entry_BA[-1][::-1]
                                        break
                    if path_to_process_BA:
                        for entry_BA in path_to_process_BA:
                            cost_potential_BA = entry_BA[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if(cost_potential_A+cost_potential_BA) == mu_AB:
                                        path_AB = potential_path_A[:-1] + entry_BA[-1][::-1]
                                        break
                    for potential_path_entry_A in Entire_path_A:
                        potential_path_A = potential_path_entry_A[-1]
                        if not(potential_path_A == '<removed-node>'):
                            cost_potential_A = potential_path_entry_A[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if(cost_potential_A+cost_potential_B) == mu_AB:
                                        path_AB = potential_path_A[:-1] + potential_path_B[::-1]
                                        break

                    if path_to_process_AC:
                        for entry_AC in path_to_process_AC:
                            cost_potential_AC = entry_AC[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if (cost_potential_AC+cost_potential_C) == mu_AC:
                                        path_AC = entry_AC[-1][:-1] + potential_path_C[::-1]
                                        break
                            if path_to_process_CA:
                                for entry_CA in path_to_process_CA:
                                    cost_potential_CA = entry_CA[0][0]
                                    if(cost_potential_AC+cost_potential_CA) == mu_AC:
                                        path_AC = entry_AC[-1][:-1] + entry_CA[-1][::-1]
                                        break
                    if path_to_process_CA:
                        for entry_CA in path_to_process_CA:
                            cost_potential_CA = entry_CA[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if(cost_potential_A+cost_potential_CA) == mu_AC:
                                        path_AC = potential_path_A[:-1] + entry_CA[-1][::-1]
                                        break
                    for potential_path_entry_A in Entire_path_A:
                        potential_path_A = potential_path_entry_A[-1]
                        if not(potential_path_A == '<removed-node>'):
                            cost_potential_A = potential_path_entry_A[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if(cost_potential_A+cost_potential_C) == mu_AC:
                                        path_AC = potential_path_A[:-1] + potential_path_C[::-1]
                                        break
                    path = path_AB[::-1][:-1] + path_AC
                    return path
                if(AB_done and BC_done) and (max(mu_AB, mu_BC) <= mu_AC):
                    print "******* A SEARCH ***************"
                    print "        "
                    print "path A"
                    path_A._print_()
                    print "    "
                    print "Path B"
                    path_B._print_()
                    print "    "
                    print "Path C"
                    path_C._print_()

                    print "        "
                    print "Path to process AB"
                    print path_to_process_AB
                    print "        "
                    print "Path to process BA"
                    print path_to_process_BA
                    print "    "
                    print "Path to process AC"
                    print path_to_process_AC
                    print "     "
                    print "Path to process CA"
                    print path_to_process_CA
                    print "    "
                    print "Path to process BC"
                    print path_to_process_BC
                    print "    "
                    print "Path to process CB"
                    print path_to_process_CB
                    print "         "
                    print "mu_AB:  ", mu_AB, "mu_AC:  ", mu_AC, "mu_BC:   ", mu_BC

                    print "**********************************"
                    Entire_path_A = path_A.copy()
                    Entire_path_B = path_B.copy()
                    Entire_path_C = path_C.copy()

                    if path_to_process_AB:
                        for entry_AB in path_to_process_AB:
                            cost_potential_AB = entry_AB[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if (cost_potential_AB+cost_potential_B) == mu_AB:
                                        path_AB = entry_AB[-1][:-1] + potential_path_B[::-1]
                                        break
                            if path_to_process_BA:
                                for entry_BA in path_to_process_BA:
                                    cost_potential_BA = entry_BA[0][0]
                                    if(cost_potential_AB+cost_potential_BA) == mu_AB:
                                        path_AB = entry_AB[-1][:-1] + entry_BA[-1][::-1]
                                        break
                    if path_to_process_BA:
                        for entry_BA in path_to_process_BA:
                            cost_potential_BA = entry_BA[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if(cost_potential_A+cost_potential_BA) == mu_AB:
                                        path_AB = potential_path_A[:-1] + entry_BA[-1][::-1]
                                        break
                    for potential_path_entry_A in Entire_path_A:
                        potential_path_A = potential_path_entry_A[-1]
                        if not(potential_path_A == '<removed-node>'):
                            cost_potential_A = potential_path_entry_A[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if(cost_potential_A+cost_potential_B) == mu_AB:
                                        path_AB = potential_path_A[:-1] + potential_path_B[::-1]
                                        break

                    if path_to_process_BC:
                        for entry_BC in path_to_process_BC:
                            cost_potential_BC = entry_BC[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if (cost_potential_BC+cost_potential_C) == mu_BC:
                                        path_BC = entry_BC[-1][:-1] + potential_path_C[::-1]
                                        break
                            if path_to_process_CB:
                                for entry_CB in path_to_process_CB:
                                    cost_potential_CB = entry_CB[0][0]
                                    if(cost_potential_BC+cost_potential_CB) == mu_BC:
                                        path_BC = entry_BC[-1][:-1] + entry_CB[-1][::-1]
                                        break
                    if path_to_process_CB:
                        for entry_CB in path_to_process_CB:
                            cost_potential_CB = entry_CB[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if(cost_potential_B+cost_potential_CB) == mu_BC:
                                        path_BC = potential_path_B[:-1] + entry_CB[-1][::-1]
                                        break
                    for potential_path_entry_B in Entire_path_B:
                        potential_path_B = potential_path_entry_B[-1]
                        if not(potential_path_B == '<removed-node>'):
                            cost_potential_B = potential_path_entry_B[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if(cost_potential_B+cost_potential_C) == mu_BC:
                                        path_BC = potential_path_B[:-1] + potential_path_C[::-1]
                                        break
                    
                    path = path_AB[:-1] + path_BC

                    print path
                    return path
            if (node_A == goals[2]) or ((top_A + top_C) > mu_AC):
                AC_done = True
                if(AC_done and AB_done) and (max(mu_AC, mu_AB) <= mu_BC):
                    print "******* A SEARCH ***************"
                    print "        "
                    print "path A"
                    path_A._print_()
                    print "    "
                    print "Path B"
                    path_B._print_()
                    print "    "
                    print "Path C"
                    path_C._print_()

                    print "        "
                    print "Path to process AB"
                    print path_to_process_AB
                    print "        "
                    print "Path to process BA"
                    print path_to_process_BA
                    print "    "
                    print "Path to process AC"
                    print path_to_process_AC
                    print "     "
                    print "Path to process CA"
                    print path_to_process_CA
                    print "    "
                    print "Path to process BC"
                    print path_to_process_BC
                    print "    "
                    print "Path to process CB"
                    print path_to_process_CB
                    print "         "
                    print "mu_AB:  ", mu_AB, "mu_AC:  ", mu_AC, "mu_BC:   ", mu_BC

                    print "**********************************"
                    Entire_path_A = path_A.copy()
                    Entire_path_B = path_B.copy()
                    Entire_path_C = path_C.copy()
                    if path_to_process_AB:
                        for entry_AB in path_to_process_AB:
                            cost_potential_AB = entry_AB[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if (cost_potential_AB+cost_potential_B) == mu_AB:
                                        path_AB = entry_AB[-1][:-1] + potential_path_B[::-1]
                                        break
                            if path_to_process_BA:
                                for entry_BA in path_to_process_BA:
                                    cost_potential_BA = entry_BA[0][0]
                                    if(cost_potential_AB+cost_potential_BA) == mu_AB:
                                        path_AB = entry_AB[-1][:-1] + entry_BA[-1][::-1]
                                        break
                    if path_to_process_BA:
                        for entry_BA in path_to_process_BA:
                            cost_potential_BA = entry_BA[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if(cost_potential_A+cost_potential_BA) == mu_AB:
                                        path_AB = potential_path_A[:-1] + entry_BA[-1][::-1]
                                        break
                    for potential_path_entry_A in Entire_path_A:
                        potential_path_A = potential_path_entry_A[-1]
                        if not(potential_path_A == '<removed-node>'):
                            cost_potential_A = potential_path_entry_A[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if(cost_potential_A+cost_potential_B) == mu_AB:
                                        path_AB = potential_path_A[:-1] + potential_path_B[::-1]
                                        break

                    if path_to_process_AC:
                        for entry_AC in path_to_process_AC:
                            cost_potential_AC = entry_AC[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if (cost_potential_AC+cost_potential_C) == mu_AC:
                                        path_AC = entry_AC[-1][:-1] + potential_path_C[::-1]
                                        break
                            if path_to_process_CA:
                                for entry_CA in path_to_process_CA:
                                    cost_potential_CA = entry_CA[0][0]
                                    if(cost_potential_AC+cost_potential_CA) == mu_AC:
                                        path_AC = entry_AC[-1][:-1] + entry_CA[-1][::-1]
                                        break
                    if path_to_process_CA:
                        for entry_CA in path_to_process_CA:
                            cost_potential_CA = entry_CA[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if(cost_potential_A+cost_potential_CA) == mu_AC:
                                        path_AC = potential_path_A[:-1] + entry_CA[-1][::-1]
                                        break
                    for potential_path_entry_A in Entire_path_A:
                        potential_path_A = potential_path_entry_A[-1]
                        if not(potential_path_A == '<removed-node>'):
                            cost_potential_A = potential_path_entry_A[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if(cost_potential_A+cost_potential_C) == mu_AC:
                                        path_AC = potential_path_A[:-1] + potential_path_C[::-1]
                                        break
                    path = path_AB[::-1][:-1] + path_AC

                    
                    
                    return path
                if(AC_done and BC_done) and(max(mu_AC, mu_BC) <= mu_AB):
                    print "******* A SEARCH ***************"
                    print "        "
                    print "path A"
                    path_A._print_()
                    print "    "
                    print "Path B"
                    path_B._print_()
                    print "    "
                    print "Path C"
                    path_C._print_()

                    print "        "
                    print "Path to process AB"
                    print path_to_process_AB
                    print "        "
                    print "Path to process BA"
                    print path_to_process_BA
                    print "    "
                    print "Path to process AC"
                    print path_to_process_AC
                    print "     "
                    print "Path to process CA"
                    print path_to_process_CA
                    print "    "
                    print "Path to process BC"
                    print path_to_process_BC
                    print "    "
                    print "Path to process CB"
                    print path_to_process_CB
                    print "         "
                    print "mu_AB:  ", mu_AB, "mu_AC:  ", mu_AC, "mu_BC:   ", mu_BC

                    print "**********************************"
                    Entire_path_A = path_A.copy()
                    Entire_path_B = path_B.copy()
                    Entire_path_C = path_C.copy()

                    if path_to_process_AC:
                        for entry_AC in path_to_process_AC:
                            cost_potential_AC = entry_AC[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if (cost_potential_AC+cost_potential_C) == mu_AC:
                                        path_AC = entry_AC[-1][:-1] + potential_path_C[::-1]
                                        break
                            if path_to_process_CA:
                                for entry_CA in path_to_process_CA:
                                    cost_potential_CA = entry_CA[0][0]
                                    if(cost_potential_AC+cost_potential_CA) == mu_AC:
                                        path_AC = entry_AC[-1][:-1] + entry_CA[-1][::-1]
                                        break
                    if path_to_process_CA:
                        for entry_CA in path_to_process_CA:
                            cost_potential_CA = entry_CA[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if(cost_potential_A+cost_potential_CA) == mu_AC:
                                        path_AB = potential_path_A[:-1] + entry_CA[-1][::-1]
                                        break
                    for potential_path_entry_A in Entire_path_A:
                        potential_path_A = potential_path_entry_A[-1]
                        if not(potential_path_A == '<removed-node>'):
                            cost_potential_A = potential_path_entry_A[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if(cost_potential_A+cost_potential_C) == mu_AC:
                                        path_AC = potential_path_A[:-1] + potential_path_C[::-1]
                                        break

                    if path_to_process_BC:
                        for entry_BC in path_to_process_BC:
                            cost_potential_BC = entry_BC[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if (cost_potential_BC+cost_potential_C) == mu_BC:
                                        path_BC = entry_BC[-1][:-1] + potential_path_C[::-1]
                                        break
                            if path_to_process_CB:
                                for entry_CB in path_to_process_CB:
                                    cost_potential_CB = entry_CB[0][0]
                                    if(cost_potential_BC+cost_potential_CB) == mu_BC:
                                        path_BC = entry_BC[-1][:-1] + entry_CB[-1][::-1]
                                        break
                    if path_to_process_CB:
                        for entry_CB in path_to_process_CB:
                            cost_potential_CB = entry_CB[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if(cost_potential_B+cost_potential_CB) == mu_BC:
                                        path_BC = potential_path_B[:-1] + entry_CB[-1][::-1]
                                        break
                    for potential_path_entry_B in Entire_path_B:
                        potential_path_B = potential_path_entry_B[-1]
                        if not(potential_path_B == '<removed-node>'):
                            cost_potential_B = potential_path_entry_B[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if(cost_potential_B+cost_potential_C) == mu_BC:
                                        path_BC = potential_path_B[:-1] + potential_path_C[::-1]
                                        break
                    path = path_AC[:-1] + path_BC[::-1]

                    
                    print path
                    return path
            explored_dict_A[node_A] = cost_A
            if count_As == 0:
                i_cost_A, i_count_A, i_path_A = path_A.pop()
                i_path_A.append(node_A)
                path_A.append((i_cost_A, i_path_A))
                count_As += 1
            parent_cost_A, parent_count_A, parent_path_A = path_A.pop()
            for child_A in graph[node_A]:
#                print "Child A"
                child_A_step_cost = graph[node_A][child_A]['weight']
                child_g_A = parent_cost_A[0] + child_A_step_cost
                if not(explored_dict_A.has_key(child_A) or frontier_A.is_node_in(child_A) or frontier_B.is_node_in(child_A) or frontier_C.is_node_in(child_A)):
                    if explored_dict_B.has_key(child_A) or explored_dict_C.has_key(child_A):
                        explored_dict_A[child_A] = child_g_A
                        if explored_dict_B.has_key(child_A):
                            path_to_process_AB.append([parent_cost_A, parent_path_A])
                            if(child_g_A + explored_dict_B[child_A]) < mu_AB:
                                mu_AB = child_g_A + explored_dict_B[child_A]
                        if explored_dict_C.has_key(child_A):
                            path_to_process_AC.append([parent_cost_A, parent_path_A])
                            if(child_g_A + explored_dict_C[child_A]) < mu_AC:
                                mu_AC = child_g_A + explored_dict_C[child_A]
                    else:
                        frontier_A.append((child_g_A, child_A))
                        temp_path_A = parent_path_A[:]
                        temp_path_A.append(child_A)
                        path_A.append(((child_g_A, child_A), temp_path_A))
                elif frontier_A.is_node_in(child_A) and (frontier_A.get_weight(child_A)[0] > child_g_A):
                    actual_cost_A, actual_count_A = frontier_A.get_weight(child_A)
                    indx_in_A = frontier_A.index([actual_cost_A, actual_count_A, child_A])
                    path_A.remove(indx_in_A)
                    frontier_A.remove_node(child_A)
                    frontier_A.append((child_g_A, child_A))
                    temp_path_A = parent_path_A[:]
                    temp_path_A.append(child_A)
                    path_A.append(((child_g_A, child_A), temp_path_A))
                    if explored_dict_B.has_key(child_A):
                        if(child_g_A + explored_dict_B[child_A]) < mu_AB:
                            mu_AB = child_g_A + explored_dict_B[child_A]
                    if explored_dict_C.has_key(child_A):
                        if(child_g_A + explored_dict_C[child_A]) < mu_AC:
                            mu_AC = child_g_A + explored_dict_C[child_A]
                            
        if not(frontier_B.is_empty()):
            cost_B, count_B, node_B = frontier_B.pop()
            top_B = cost_B
            if(node_B == goals[0]) or ((top_B + top_A) > mu_AB):
                AB_done = True
                if(AB_done and AC_done) and (max(mu_AB, mu_AC) <= mu_BC):
                    print "******* B SEARCH ***************"
                    print "        "
                    print "path A"
                    path_A._print_()
                    print "    "
                    print "Path B"
                    path_B._print_()
                    print "    "
                    print "Path C"
                    path_C._print_()

                    print "        "
                    print "Path to process AB"
                    print path_to_process_AB
                    print "        "
                    print "Path to process BA"
                    print path_to_process_BA
                    print "    "
                    print "Path to process AC"
                    print path_to_process_AC
                    print "     "
                    print "Path to process CA"
                    print path_to_process_CA
                    print "    "
                    print "Path to process BC"
                    print path_to_process_BC
                    print "    "
                    print "Path to process CB"
                    print path_to_process_CB
                    print "         "
                    print "mu_AB:  ", mu_AB, "mu_AC:  ", mu_AC, "mu_BC:   ", mu_BC

                    print "**********************************"
                    Entire_path_A = path_A.copy()
                    Entire_path_B = path_B.copy()
                    Entire_path_C = path_C.copy()
                    if path_to_process_AB:
                        for entry_AB in path_to_process_AB:
                            cost_potential_AB = entry_AB[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if (cost_potential_AB+cost_potential_B) == mu_AB:
                                        path_AB = entry_AB[-1][:-1] + potential_path_B[::-1]
                                        break
                            if path_to_process_BA:
                                for entry_BA in path_to_process_BA:
                                    cost_potential_BA = entry_BA[0][0]
                                    if(cost_potential_AB+cost_potential_BA) == mu_AB:
                                        path_AB = entry_AB[-1][:-1] + entry_BA[-1][::-1]
                                        break
                    if path_to_process_BA:
                        for entry_BA in path_to_process_BA:
                            cost_potential_BA = entry_BA[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if(cost_potential_A+cost_potential_BA) == mu_AB:
                                        path_AB = potential_path_A[:-1] + entry_BA[-1][::-1]
                                        break
                    for potential_path_entry_A in Entire_path_A:
                        potential_path_A = potential_path_entry_A[-1]
                        if not(potential_path_A == '<removed-node>'):
                            cost_potential_A = potential_path_entry_A[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if(cost_potential_A+cost_potential_B) == mu_AB:
                                        path_AB = potential_path_A[:-1] + potential_path_B[::-1]
                                        break

                    if path_to_process_AC:
                        for entry_AC in path_to_process_AC:
                            cost_potential_AC = entry_AC[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if (cost_potential_AC+cost_potential_C) == mu_AC:
                                        path_AC = entry_AC[-1][:-1] + potential_path_C[::-1]
                                        break
                            if path_to_process_CA:
                                for entry_CA in path_to_process_CA:
                                    cost_potential_CA = entry_CA[0][0]
                                    if(cost_potential_AC+cost_potential_CA) == mu_AC:
                                        path_AC = entry_AC[-1][:-1] + entry_CA[-1][::-1]
                                        break
                    if path_to_process_CA:
                        for entry_CA in path_to_process_CA:
                            cost_potential_CA = entry_CA[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if(cost_potential_A+cost_potential_CA) == mu_AC:
                                        path_AC = potential_path_A[:-1] + entry_CA[-1][::-1]
                                        break
                    for potential_path_entry_A in Entire_path_A:
                        potential_path_A = potential_path_entry_A[-1]
                        if not(potential_path_A == '<removed-node>'):
                            cost_potential_A = potential_path_entry_A[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if(cost_potential_A+cost_potential_C) == mu_AC:
                                        path_AC = potential_path_A[:-1] + potential_path_C[::-1]
                                        break
                    path = path_AB[::-1][:-1] + path_AC

                    
                    return path
                if(AB_done and BC_done) and (max(mu_AB, mu_BC) <= mu_AC):
                    print "******* B SEARCH ***************"
                    print "        "
                    print "path A"
                    path_A._print_()
                    print "    "
                    print "Path B"
                    path_B._print_()
                    print "    "
                    print "Path C"
                    path_C._print_()

                    print "        "
                    print "Path to process AB"
                    print path_to_process_AB
                    print "        "
                    print "Path to process BA"
                    print path_to_process_BA
                    print "    "
                    print "Path to process AC"
                    print path_to_process_AC
                    print "     "
                    print "Path to process CA"
                    print path_to_process_CA
                    print "    "
                    print "Path to process BC"
                    print path_to_process_BC
                    print "    "
                    print "Path to process CB"
                    print path_to_process_CB
                    print "         "
                    print "mu_AB:  ", mu_AB, "mu_AC:  ", mu_AC, "mu_BC:   ", mu_BC

                    print "**********************************"
                    Entire_path_A = path_A.copy()
                    Entire_path_B = path_B.copy()
                    Entire_path_C = path_C.copy()

                    if path_to_process_AB:
                        for entry_AB in path_to_process_AB:
                            cost_potential_AB = entry_AB[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if (cost_potential_AB+cost_potential_B) == mu_AB:
                                        path_AB = entry_AB[-1][:-1] + potential_path_B[::-1]
                                        break
                            if path_to_process_BA:
                                for entry_BA in path_to_process_BA:
                                    cost_potential_BA = entry_BA[0][0]
                                    if(cost_potential_AB+cost_potential_BA) == mu_AB:
                                        path_AB = entry_AB[-1][:-1] + entry_BA[-1][::-1]
                                        break
                    if path_to_process_BA:
                        for entry_BA in path_to_process_BA:
                            cost_potential_BA = entry_BA[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if(cost_potential_A+cost_potential_BA) == mu_AB:
                                        path_AB = potential_path_A[:-1] + entry_BA[-1][::-1]
                                        break
                    for potential_path_entry_A in Entire_path_A:
                        potential_path_A = potential_path_entry_A[-1]
                        if not(potential_path_A == '<removed-node>'):
                            cost_potential_A = potential_path_entry_A[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if(cost_potential_A+cost_potential_B) == mu_AB:
                                        path_AB = potential_path_A[:-1] + potential_path_B[::-1]
                                        break

                    if path_to_process_BC:
                        for entry_BC in path_to_process_BC:
                            cost_potential_BC = entry_BC[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if (cost_potential_BC+cost_potential_C) == mu_BC:
                                        path_BC = entry_BC[-1][:-1] + potential_path_C[::-1]
                                        break
                            if path_to_process_CB:
                                for entry_CB in path_to_process_CB:
                                    cost_potential_CB = entry_CB[0][0]
                                    if(cost_potential_BC+cost_potential_CB) == mu_BC:
                                        path_BC = entry_BC[-1][:-1] + entry_CB[-1][::-1]
                                        break
                    if path_to_process_CB:
                        for entry_CB in path_to_process_CB:
                            cost_potential_CB = entry_CB[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if(cost_potential_B+cost_potential_CB) == mu_BC:
                                        path_BC = potential_path_B[:-1] + entry_CB[-1][::-1]
                                        break
                    for potential_path_entry_B in Entire_path_B:
                        potential_path_B = potential_path_entry_B[-1]
                        if not(potential_path_B == '<removed-node>'):
                            cost_potential_B = potential_path_entry_B[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if(cost_potential_B+cost_potential_C) == mu_BC:
                                        path_BC = potential_path_B[:-1] + potential_path_C[::-1]
                                        break
                    path = path_AB[:-1] + path_BC


                    
                    return path
            if(node_B == goals[2]) or ((top_B + top_C) > mu_BC):
                BC_done = True
                if(BC_done and AB_done) and (max(mu_BC, mu_AB) <= mu_AC):
                    print "******* B SEARCH ***************"
                    print "        "
                    print "path A"
                    path_A._print_()
                    print "    "
                    print "Path B"
                    path_B._print_()
                    print "    "
                    print "Path C"
                    path_C._print_()

                    print "        "
                    print "Path to process AB"
                    print path_to_process_AB
                    print "        "
                    print "Path to process BA"
                    print path_to_process_BA
                    print "    "
                    print "Path to process AC"
                    print path_to_process_AC
                    print "     "
                    print "Path to process CA"
                    print path_to_process_CA
                    print "    "
                    print "Path to process BC"
                    print path_to_process_BC
                    print "    "
                    print "Path to process CB"
                    print path_to_process_CB
                    print "         "
                    print "mu_AB:  ", mu_AB, "mu_AC:  ", mu_AC, "mu_BC:   ", mu_BC

                    print "**********************************"
                    Entire_path_A = path_A.copy()
                    Entire_path_B = path_B.copy()
                    Entire_path_C = path_C.copy()

                    if path_to_process_AB:
                        for entry_AB in path_to_process_AB:
                            cost_potential_AB = entry_AB[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if (cost_potential_AB+cost_potential_B) == mu_AB:
                                        path_AB = entry_AB[-1][:-1] + potential_path_B[::-1]
                                        break
                            if path_to_process_BA:
                                for entry_BA in path_to_process_BA:
                                    cost_potential_BA = entry_BA[0][0]
                                    if(cost_potential_AB+cost_potential_BA) == mu_AB:
                                        path_AB = entry_AB[-1][:-1] + entry_BA[-1][::-1]
                                        break
                    if path_to_process_BA:
                        for entry_BA in path_to_process_BA:
                            cost_potential_BA = entry_BA[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if(cost_potential_A+cost_potential_BA) == mu_AB:
                                        path_AB = potential_path_A[:-1] + entry_BA[-1][::-1]
                                        break
                    for potential_path_entry_A in Entire_path_A:
                        potential_path_A = potential_path_entry_A[-1]
                        if not(potential_path_A == '<removed-node>'):
                            cost_potential_A = potential_path_entry_A[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if(cost_potential_A+cost_potential_B) == mu_AB:
                                        path_AB = potential_path_A[:-1] + potential_path_B[::-1]
                                        break

                    if path_to_process_BC:
                        for entry_BC in path_to_process_BC:
                            cost_potential_BC = entry_BC[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if (cost_potential_BC+cost_potential_C) == mu_BC:
                                        path_BC = entry_BC[-1][:-1] + potential_path_C[::-1]
                                        break
                            if path_to_process_CB:
                                for entry_CB in path_to_process_CB:
                                    cost_potential_CB = entry_CB[0][0]
                                    if(cost_potential_BC+cost_potential_CB) == mu_BC:
                                        path_BC = entry_BC[-1][:-1] + entry_CB[-1][::-1]
                                        break
                    if path_to_process_CB:
                        for entry_CB in path_to_process_CB:
                            cost_potential_CB = entry_CB[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if(cost_potential_B+cost_potential_CB) == mu_BC:
                                        path_BC = potential_path_B[:-1] + entry_CB[-1][::-1]
                                        break
                    for potential_path_entry_B in Entire_path_B:
                        potential_path_B = potential_path_entry_B[-1]
                        if not(potential_path_B == '<removed-node>'):
                            cost_potential_B = potential_path_entry_B[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if(cost_potential_B+cost_potential_C) == mu_BC:
                                        path_BC = potential_path_B[:-1] + potential_path_C[::-1]
                                        break
                    path = path_AB[:-1] + path_BC

                    
                    return path
                if(BC_done and AC_done) and (max(mu_BC, mu_AC) <= mu_AB):
                    print "******* B SEARCH ***************"
                    print "        "
                    print "path A"
                    path_A._print_()
                    print "    "
                    print "Path B"
                    path_B._print_()
                    print "    "
                    print "Path C"
                    path_C._print_()

                    print "        "
                    print "Path to process AB"
                    print path_to_process_AB
                    print "        "
                    print "Path to process BA"
                    print path_to_process_BA
                    print "    "
                    print "Path to process AC"
                    print path_to_process_AC
                    print "     "
                    print "Path to process CA"
                    print path_to_process_CA
                    print "    "
                    print "Path to process BC"
                    print path_to_process_BC
                    print "    "
                    print "Path to process CB"
                    print path_to_process_CB
                    print "         "
                    print "mu_AB:  ", mu_AB, "mu_AC:  ", mu_AC, "mu_BC:   ", mu_BC

                    print "**********************************"
                    Entire_path_A = path_A.copy()
                    Entire_path_B = path_B.copy()
                    Entire_path_C = path_C.copy()

                    if path_to_process_AC:
                        for entry_AC in path_to_process_AC:
                            cost_potential_AC = entry_AC[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if (cost_potential_AC+cost_potential_C) == mu_AC:
                                        path_AC = entry_AC[-1][:-1] + potential_path_C[::-1]
                                        break
                            if path_to_process_CA:
                                for entry_CA in path_to_process_CA:
                                    cost_potential_CA = entry_CA[0][0]
                                    if(cost_potential_AC+cost_potential_CA) == mu_AC:
                                        path_AC = entry_AC[-1][:-1] + entry_CA[-1][::-1]
                                        break
                    if path_to_process_CA:
                        for entry_CA in path_to_process_CA:
                            cost_potential_CA = entry_CA[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if(cost_potential_A+cost_potential_CA) == mu_AC:
                                        path_AB = potential_path_A[:-1] + entry_CA[-1][::-1]
                                        break
                    for potential_path_entry_A in Entire_path_A:
                        potential_path_A = potential_path_entry_A[-1]
                        if not(potential_path_A == '<removed-node>'):
                            cost_potential_A = potential_path_entry_A[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if(cost_potential_A+cost_potential_C) == mu_AC:
                                        path_AC = potential_path_A[:-1] + potential_path_C[::-1]
                                        break

                    if path_to_process_BC:
                        for entry_BC in path_to_process_BC:
                            cost_potential_BC = entry_BC[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if (cost_potential_BC+cost_potential_C) == mu_BC:
                                        path_BC = entry_BC[-1][:-1] + potential_path_C[::-1]
                                        break
                            if path_to_process_CB:
                                for entry_CB in path_to_process_CB:
                                    cost_potential_CB = entry_CB[0][0]
                                    if(cost_potential_BC+cost_potential_CB) == mu_BC:
                                        path_BC = entry_BC[-1][:-1] + entry_CB[-1][::-1]
                                        break
                    if path_to_process_CB:
                        for entry_CB in path_to_process_CB:
                            cost_potential_CB = entry_CB[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if(cost_potential_B+cost_potential_CB) == mu_BC:
                                        path_BC = potential_path_B[:-1] + entry_CB[-1][::-1]
                                        break
                    for potential_path_entry_B in Entire_path_B:
                        potential_path_B = potential_path_entry_B[-1]
                        if not(potential_path_B == '<removed-node>'):
                            cost_potential_B = potential_path_entry_B[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if(cost_potential_B+cost_potential_C) == mu_BC:
                                        path_BC = potential_path_B[:-1] + potential_path_C[::-1]
                                        break
                    path = path_AC[:-1] + path_BC[::-1]

                    
                    print path

                    
                    return path
            explored_dict_B[node_B] = cost_B
            if count_Bs == 0:
                i_cost_B, i_count_B, i_path_B = path_B.pop()
                i_path_B.append(node_B)
                path_B.append((i_cost_B, i_path_B))
                count_Bs += 1
            parent_cost_B, parent_count_B, parent_path_B = path_B.pop()
            for child_B in graph[node_B]:
                print "Parent", node_B
                print "child B", child_B
                child_B_step_cost = graph[node_B][child_B]['weight']
                child_g_B = parent_cost_B[0] + child_B_step_cost
                if not(explored_dict_B.has_key(child_B) or frontier_B.is_node_in(child_B) or frontier_C.is_node_in(child_B) or frontier_A.is_node_in(child_B)):
                    if explored_dict_A.has_key(child_B) or explored_dict_C.has_key(child_B):
                        explored_dict_B[child_B] = child_g_B
                        if explored_dict_A.has_key(child_B):
                            path_to_process_BA.append([parent_cost_B, parent_path_B])
                            print "Path to process BA"
                            print path_to_process_BA
                            if(child_g_B + explored_dict_A[child_B]) < mu_AB:
                                mu_AB = child_g_B + explored_dict_A[child_B]
                        if explored_dict_C.has_key(child_B):
                            path_to_process_BC.append([parent_cost_B, parent_path_B])
                            print "Path to process BC"
                            print path_to_process_BC
                            if(child_g_B + explored_dict_C[child_B]) < mu_BC:
                                mu_BC = child_g_B + explored_dict_C[child_B]
                    else:
                        frontier_B.append((child_g_B, child_B))
                        print "Frontier B"
                        frontier_B._print_()
                        print "frontier A"
                        frontier_A._print_()
                        print "frontier C"
                        frontier_C._print_()
                        temp_path_B = parent_path_B[:]
                        temp_path_B.append(child_B)
                        path_B.append(((child_g_B, child_B), temp_path_B))
                        print "Path B"
                        path_B._print_()
#                        print "path B in loop"
#                        path_B._print_()
#                        print "frontier B in loop"
#                        frontier_B._print_()

                elif frontier_B.is_node_in(child_B) and (frontier_B.get_weight(child_B)[0] > child_g_B):
                    actual_cost_B, actual_count_B = frontier_B.get_weight(child_B)
                    indx_in_B = frontier_B.index([actual_cost_B, actual_count_B, child_B])
                    path_B.remove(indx_in_B)
                    frontier_B.remove_node(child_B)
                    frontier_B.append((child_g_B, child_B))
#                    print "REPLACED FRONTIER IN B"
                    frontier_B._print_()
                    temp_path_B = parent_path_B[:]
                    temp_path_B.append(child_B)
                    path_B.append(((child_g_B, child_B), temp_path_B))
#                    print "REPLACED PATH IN B"
                    path_B._print_()
                    if explored_dict_A.has_key(child_B):
                        if(child_g_B + explored_dict_A[child_B]) < mu_AB:
                            mu_AB = child_g_B + explored_dict_A[child_B]
                    if explored_dict_C.has_key(child_B):
                        if(child_g_B + explored_dict_C[child_B]) < mu_BC:
                            mu_BC = child_g_B + explored_dict_C[child_B]
                            
        if not(frontier_C.is_empty()):
            cost_C, count_C, node_C = frontier_C.pop()
            top_C = cost_C
            if (node_C == goals[0]) or ((top_C + top_A) > mu_AC):
                AC_done = True
                if(AC_done and AB_done) and (max(mu_AC, mu_AB) <= mu_BC):
                    print "******* C SEARCH ***************"
                    print "        "
                    print "path A"
                    path_A._print_()
                    print "    "
                    print "Path B"
                    path_B._print_()
                    print "    "
                    print "Path C"
                    path_C._print_()

                    print "        "
                    print "Path to process AB"
                    print path_to_process_AB
                    print "        "
                    print "Path to process BA"
                    print path_to_process_BA
                    print "    "
                    print "Path to process AC"
                    print path_to_process_AC
                    print "     "
                    print "Path to process CA"
                    print path_to_process_CA
                    print "    "
                    print "Path to process BC"
                    print path_to_process_BC
                    print "    "
                    print "Path to process CB"
                    print path_to_process_CB
                    print "         "
                    print "mu_AB:  ", mu_AB, "mu_AC:  ", mu_AC, "mu_BC:   ", mu_BC

                    print "**********************************"
                    Entire_path_A = path_A.copy()
                    Entire_path_B = path_B.copy()
                    Entire_path_C = path_C.copy()
                    if path_to_process_AB:
                        for entry_AB in path_to_process_AB:
                            cost_potential_AB = entry_AB[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if (cost_potential_AB+cost_potential_B) == mu_AB:
                                        path_AB = entry_AB[-1][:-1] + potential_path_B[::-1]
                                        break
                            if path_to_process_BA:
                                for entry_BA in path_to_process_BA:
                                    cost_potential_BA = entry_BA[0][0]
                                    if(cost_potential_AB+cost_potential_BA) == mu_AB:
                                        path_AB = entry_AB[-1][:-1] + entry_BA[-1][::-1]
                                        break
                    if path_to_process_BA:
                        for entry_BA in path_to_process_BA:
                            cost_potential_BA = entry_BA[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if(cost_potential_A+cost_potential_BA) == mu_AB:
                                        path_AB = potential_path_A[:-1] + entry_BA[-1][::-1]
                                        break
                    for potential_path_entry_A in Entire_path_A:
                        potential_path_A = potential_path_entry_A[-1]
                        if not(potential_path_A == '<removed-node>'):
                            cost_potential_A = potential_path_entry_A[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if(cost_potential_A+cost_potential_B) == mu_AB:
                                        path_AB = potential_path_A[:-1] + potential_path_B[::-1]
                                        break

                    if path_to_process_AC:
                        for entry_AC in path_to_process_AC:
                            cost_potential_AC = entry_AC[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if (cost_potential_AC+cost_potential_C) == mu_AC:
                                        path_AC = entry_AC[-1][:-1] + potential_path_C[::-1]
                                        break
                            if path_to_process_CA:
                                for entry_CA in path_to_process_CA:
                                    cost_potential_CA = entry_CA[0][0]
                                    if(cost_potential_AC+cost_potential_CA) == mu_AC:
                                        path_AC = entry_AC[-1][:-1] + entry_CA[-1][::-1]
                                        break
                    if path_to_process_CA:
                        for entry_CA in path_to_process_CA:
                            cost_potential_CA = entry_CA[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if(cost_potential_A+cost_potential_CA) == mu_AC:
                                        path_AC = potential_path_A[:-1] + entry_CA[-1][::-1]
                                        break
                    for potential_path_entry_A in Entire_path_A:
                        potential_path_A = potential_path_entry_A[-1]
                        if not(potential_path_A == '<removed-node>'):
                            cost_potential_A = potential_path_entry_A[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if(cost_potential_A+cost_potential_C) == mu_AC:
                                        path_AC = potential_path_A[:-1] + potential_path_C[::-1]
                                        break
                    path = path_AB[::-1][:-1] + path_AC


                    
                    return path
                if(AC_done and BC_done) and (max(mu_AC, mu_BC) <= mu_AB):
                    print "******* C SEARCH ***************"
                    print "        "
                    print "path A"
                    path_A._print_()
                    print "    "
                    print "Path B"
                    path_B._print_()
                    print "    "
                    print "Path C"
                    path_C._print_()

                    print "        "
                    print "Path to process AB"
                    print path_to_process_AB
                    print "        "
                    print "Path to process BA"
                    print path_to_process_BA
                    print "    "
                    print "Path to process AC"
                    print path_to_process_AC
                    print "     "
                    print "Path to process CA"
                    print path_to_process_CA
                    print "    "
                    print "Path to process BC"
                    print path_to_process_BC
                    print "    "
                    print "Path to process CB"
                    print path_to_process_CB
                    print "         "
                    print "mu_AB:  ", mu_AB, "mu_AC:  ", mu_AC, "mu_BC:   ", mu_BC

                    print "**********************************"
                    Entire_path_A = path_A.copy()
                    Entire_path_B = path_B.copy()
                    Entire_path_C = path_C.copy()

                    if path_to_process_AC:
                        for entry_AC in path_to_process_AC:
                            cost_potential_AC = entry_AC[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if (cost_potential_AC+cost_potential_C) == mu_AC:
                                        path_AC = entry_AC[-1][:-1] + potential_path_C[::-1]
                                        break
                            if path_to_process_CA:
                                for entry_CA in path_to_process_CA:
                                    cost_potential_CA = entry_CA[0][0]
                                    if(cost_potential_AC+cost_potential_CA) == mu_AC:
                                        path_AC = entry_AC[-1][:-1] + entry_CA[-1][::-1]
                                        break
                    if path_to_process_CA:
                        for entry_CA in path_to_process_CA:
                            cost_potential_CA = entry_CA[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if(cost_potential_A+cost_potential_CA) == mu_AC:
                                        path_AB = potential_path_A[:-1] + entry_CA[-1][::-1]
                                        break
                    for potential_path_entry_A in Entire_path_A:
                        potential_path_A = potential_path_entry_A[-1]
                        if not(potential_path_A == '<removed-node>'):
                            cost_potential_A = potential_path_entry_A[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if(cost_potential_A+cost_potential_C) == mu_AC:
                                        path_AC = potential_path_A[:-1] + potential_path_C[::-1]
                                        break

                    if path_to_process_BC:
                        for entry_BC in path_to_process_BC:
                            cost_potential_BC = entry_BC[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if (cost_potential_BC+cost_potential_C) == mu_BC:
                                        path_BC = entry_BC[-1][:-1] + potential_path_C[::-1]
                                        break
                            if path_to_process_CB:
                                for entry_CB in path_to_process_CB:
                                    cost_potential_CB = entry_CB[0][0]
                                    if(cost_potential_BC+cost_potential_CB) == mu_BC:
                                        path_BC = entry_BC[-1][:-1] + entry_CB[-1][::-1]
                                        break
                    if path_to_process_CB:
                        for entry_CB in path_to_process_CB:
                            cost_potential_CB = entry_CB[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if(cost_potential_B+cost_potential_CB) == mu_BC:
                                        path_BC = potential_path_B[:-1] + entry_CB[-1][::-1]
                                        break
                    for potential_path_entry_B in Entire_path_B:
                        potential_path_B = potential_path_entry_B[-1]
                        if not(potential_path_B == '<removed-node>'):
                            cost_potential_B = potential_path_entry_B[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if(cost_potential_B+cost_potential_C) == mu_BC:
                                        path_BC = potential_path_B[:-1] + potential_path_C[::-1]
                                        break
                    path = path_AC[:-1] + path_BC[::-1]

                    
                    print path

                    
                    return path
            if (node_C == goals[1]) or ((top_C + top_B) > mu_BC):
                BC_done = True
                if(BC_done and AB_done) and (max(mu_BC, mu_AB) <= mu_AC):
                    print "******* C SEARCH ***************"
                    print "        "
                    print "path A"
                    path_A._print_()
                    print "    "
                    print "Path B"
                    path_B._print_()
                    print "    "
                    print "Path C"
                    path_C._print_()

                    print "        "
                    print "Path to process AB"
                    print path_to_process_AB
                    print "        "
                    print "Path to process BA"
                    print path_to_process_BA
                    print "    "
                    print "Path to process AC"
                    print path_to_process_AC
                    print "     "
                    print "Path to process CA"
                    print path_to_process_CA
                    print "    "
                    print "Path to process BC"
                    print path_to_process_BC
                    print "    "
                    print "Path to process CB"
                    print path_to_process_CB
                    print "         "
                    print "mu_AB:  ", mu_AB, "mu_AC:  ", mu_AC, "mu_BC:   ", mu_BC

                    print "**********************************"
                    Entire_path_A = path_A.copy()
                    Entire_path_B = path_B.copy()
                    Entire_path_C = path_C.copy()

                    if path_to_process_AB:
                        for entry_AB in path_to_process_AB:
                            cost_potential_AB = entry_AB[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if (cost_potential_AB+cost_potential_B) == mu_AB:
                                        path_AB = entry_AB[-1][:-1] + potential_path_B[::-1]
                                        break
                            if path_to_process_BA:
                                for entry_BA in path_to_process_BA:
                                    cost_potential_BA = entry_BA[0][0]
                                    if(cost_potential_AB+cost_potential_BA) == mu_AB:
                                        path_AB = entry_AB[-1][:-1] + entry_BA[-1][::-1]
                                        break
                    if path_to_process_BA:
                        for entry_BA in path_to_process_BA:
                            cost_potential_BA = entry_BA[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if(cost_potential_A+cost_potential_BA) == mu_AB:
                                        path_AB = potential_path_A[:-1] + entry_BA[-1][::-1]
                                        break
                    for potential_path_entry_A in Entire_path_A:
                        potential_path_A = potential_path_entry_A[-1]
                        if not(potential_path_A == '<removed-node>'):
                            cost_potential_A = potential_path_entry_A[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if(cost_potential_A+cost_potential_B) == mu_AB:
                                        path_AB = potential_path_A[:-1] + potential_path_B[::-1]
                                        break

                    if path_to_process_BC:
                        for entry_BC in path_to_process_BC:
                            cost_potential_BC = entry_BC[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if (cost_potential_BC+cost_potential_C) == mu_BC:
                                        path_BC = entry_BC[-1][:-1] + potential_path_C[::-1]
                                        break
                            if path_to_process_CB:
                                for entry_CB in path_to_process_CB:
                                    cost_potential_CB = entry_CB[0][0]
                                    if(cost_potential_BC+cost_potential_CB) == mu_BC:
                                        path_BC = entry_BC[-1][:-1] + entry_CB[-1][::-1]
                                        break
                    if path_to_process_CB:
                        for entry_CB in path_to_process_CB:
                            cost_potential_CB = entry_CB[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if(cost_potential_B+cost_potential_CB) == mu_BC:
                                        path_BC = potential_path_B[:-1] + entry_CB[-1][::-1]
                                        break
                    for potential_path_entry_B in Entire_path_B:
                        potential_path_B = potential_path_entry_B[-1]
                        if not(potential_path_B == '<removed-node>'):
                            cost_potential_B = potential_path_entry_B[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if(cost_potential_B+cost_potential_C) == mu_BC:
                                        path_BC = potential_path_B[:-1] + potential_path_C[::-1]
                                        break
                    path = path_AB[:-1] + path_BC


                    return path
                if(BC_done and AC_done) and (max(mu_BC, mu_AC) <= mu_AB):
                    print "******* A SEARCH ***************"
                    print "        "
                    print "path A"
                    path_A._print_()
                    print "    "
                    print "Path B"
                    path_B._print_()
                    print "    "
                    print "Path C"
                    path_C._print_()

                    print "        "
                    print "Path to process AB"
                    print path_to_process_AB
                    print "        "
                    print "Path to process BA"
                    print path_to_process_BA
                    print "    "
                    print "Path to process AC"
                    print path_to_process_AC
                    print "     "
                    print "Path to process CA"
                    print path_to_process_CA
                    print "    "
                    print "Path to process BC"
                    print path_to_process_BC
                    print "    "
                    print "Path to process CB"
                    print path_to_process_CB
                    print "         "
                    print "mu_AB:  ", mu_AB, "mu_AC:  ", mu_AC, "mu_BC:   ", mu_BC

                    print "**********************************"
                    Entire_path_A = path_A.copy()
                    Entire_path_B = path_B.copy()
                    Entire_path_C = path_C.copy()

                    if path_to_process_AC:
                        for entry_AC in path_to_process_AC:
                            cost_potential_AC = entry_AC[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if (cost_potential_AC+cost_potential_C) == mu_AC:
                                        path_AC = entry_AC[-1][:-1] + potential_path_C[::-1]
                                        break
                            if path_to_process_CA:
                                for entry_CA in path_to_process_CA:
                                    cost_potential_CA = entry_CA[0][0]
                                    if(cost_potential_AC+cost_potential_CA) == mu_AC:
                                        path_AC = entry_AC[-1][:-1] + entry_CA[-1][::-1]
                                        break
                    if path_to_process_CA:
                        for entry_CA in path_to_process_CA:
                            cost_potential_CA = entry_CA[0][0]
                            for potential_path_entry_A in Entire_path_A:
                                potential_path_A = potential_path_entry_A[-1]
                                if not(potential_path_A == '<removed-node>'):
                                    cost_potential_A = potential_path_entry_A[0][0]
                                    if(cost_potential_A+cost_potential_CA) == mu_AC:
                                        path_AB = potential_path_A[:-1] + entry_CA[-1][::-1]
                                        break
                    for potential_path_entry_A in Entire_path_A:
                        potential_path_A = potential_path_entry_A[-1]
                        if not(potential_path_A == '<removed-node>'):
                            cost_potential_A = potential_path_entry_A[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if(cost_potential_A+cost_potential_C) == mu_AC:
                                        path_AC = potential_path_A[:-1] + potential_path_C[::-1]
                                        break

                    if path_to_process_BC:
                        for entry_BC in path_to_process_BC:
                            cost_potential_BC = entry_BC[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if (cost_potential_BC+cost_potential_C) == mu_BC:
                                        path_BC = entry_BC[-1][:-1] + potential_path_C[::-1]
                                        break
                            if path_to_process_CB:
                                for entry_CB in path_to_process_CB:
                                    cost_potential_CB = entry_CB[0][0]
                                    if(cost_potential_BC+cost_potential_CB) == mu_BC:
                                        path_BC = entry_BC[-1][:-1] + entry_CB[-1][::-1]
                                        break
                    if path_to_process_CB:
                        for entry_CB in path_to_process_CB:
                            cost_potential_CB = entry_CB[0][0]
                            for potential_path_entry_B in Entire_path_B:
                                potential_path_B = potential_path_entry_B[-1]
                                if not(potential_path_B == '<removed-node>'):
                                    cost_potential_B = potential_path_entry_B[0][0]
                                    if(cost_potential_B+cost_potential_CB) == mu_BC:
                                        path_BC = potential_path_B[:-1] + entry_CB[-1][::-1]
                                        break
                    for potential_path_entry_B in Entire_path_B:
                        potential_path_B = potential_path_entry_B[-1]
                        if not(potential_path_B == '<removed-node>'):
                            cost_potential_B = potential_path_entry_B[0][0]
                            for potential_path_entry_C in Entire_path_C:
                                potential_path_C = potential_path_entry_C[-1]
                                if not(potential_path_C == '<removed-node>'):
                                    cost_potential_C = potential_path_entry_C[0][0]
                                    if(cost_potential_B+cost_potential_C) == mu_BC:
                                        path_BC = potential_path_B[:-1] + potential_path_C[::-1]
                                        break
                    path = path_AC[:-1] + path_BC[::-1]

                    
                    print path

                    
                    return path
            explored_dict_C[node_C] = cost_C
            if count_Cs == 0:
                i_cost_C, i_count_C, i_path_C = path_C.pop()
                i_path_C.append(node_C)
                path_C.append((i_cost_C, i_path_C))
                count_Cs += 1
            parent_cost_C, parent_count_C, parent_path_C = path_C.pop()
            for child_C in graph[node_C]:
                child_C_step_cost = graph[node_C][child_C]['weight']
                child_g_C = parent_cost_C[0] + child_C_step_cost
                if not(explored_dict_C.has_key(child_C) or frontier_C.is_node_in(child_C) or frontier_A.is_node_in(child_C) or frontier_B.is_node_in(child_C)):
                    if explored_dict_A.has_key(child_C) or explored_dict_B.has_key(child_C):
                        explored_dict_C[child_C] = child_g_C
                        if explored_dict_A.has_key(child_C):
                            path_to_process_CA.append([parent_cost_C, parent_path_C])
                            if(child_g_C + explored_dict_A[child_C]) < mu_AC:
                                mu_AC = child_g_C + explored_dict_A[child_C]
                        if explored_dict_B.has_key(child_C):
                            path_to_process_CB.append([parent_cost_C, parent_path_C])
                            if(child_g_C + explored_dict_B[child_C]) < mu_BC:
                                mu_BC = child_g_C + explored_dict_B[child_C]
                    else:
                        frontier_C.append((child_g_C, child_C))
                        temp_path_C = parent_path_C[:]
                        temp_path_C.append(child_C)
                        path_C.append(((child_g_C, child_C), temp_path_C))
                elif frontier_C.is_node_in(child_C) and (frontier_C.get_weight(child_C)[0] > child_g_C):
                    actual_cost_C, actual_count_C = frontier_C.get_weight(child_C)
                    indx_in_C = frontier_C.index([actual_cost_C, actual_count_C, child_C])
                    path_C.remove(indx_in_C)
                    frontier_C.remove_node(child_C)
                    frontier_C.append((child_g_C, child_C))
                    temp_path_C = parent_path_C[:]
                    temp_path_C.append(child_C)
                    path_C.append(((child_g_C, child_C), temp_path_C))
                    if explored_dict_A.has_key(child_C):
                        if(child_g_C + explored_dict_A[child_C]) < mu_AC:
                            mu_AC = child_g_C + explored_dict_A[child_C]
                    if explored_dict_B.has_key(child_C):
                        if(child_g_C + explored_dict_B[child_C]) < mu_BC:
                            mu_BC = child_g_C + explored_dict_B[child_C]


                    
def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    """
    Exercise 3: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    raise NotImplementedError


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Dan Monga Kilanga"


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data():
    """
    Loads data from data.pickle and return the data object that is passed to
    the custom_search method.

    Will be called only once. Feel free to modify.

    Returns:
         The data loaded from the pickle file.
    """

    dir_name = os.path.dirname(os.path.realpath(__file__))
    pickle_file_path = os.path.join(dir_name, "data.pickle")
    data = pickle.load(open(pickle_file_path, 'rb'))
    return data
