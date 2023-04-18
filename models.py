from typing import Tuple
# import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import random


class Graph:

    def __init__(self, factor: int) -> None:
        self.factor = factor

    @staticmethod
    def _generate(n: int, factor: float) -> list[list[int]]:
        possible_edges = n*(n-1)/2
        fill_edges = math.floor(factor * possible_edges) - n
        a_mat = [[0 for _ in range(n)] for _ in range(n)]

        #make sure graph will be connected
        nodes = [i for i in range(n)]
        np.random.shuffle(nodes)
        last = nodes[0]
        for node in nodes[1:]:
            from_, to = min(last, node), max(last, node)
            a_mat[from_][to] = 1


        #fill rest of edges
        for i in range(fill_edges):
            while True:
                from_ = random.randint(0,n-1)
                to = random.randint(from_,  n-1)
                if from_ == to:
                    continue
                if not a_mat[from_][to]:
                    break
            a_mat[from_][to] = 1
        return a_mat

    def _input_graph(self) -> bool | list[list[int]]:
        a_mat = []
        n = int(input("Set size [int]: "))
        print("input adjacency matrix:")
        for i in range(n):
            try:
                temp = list(map(int, input().split()))
                if len(temp) != n:
                    print("Invalid size.")
                    return False
                a_mat.append(temp)
            except ValueError:
                print("One or more not integer values.")
                return False

        return a_mat

    @staticmethod
    def a_mat2a_list(a_mat: list[list[int]]) -> list[int]:
        a_list = []
        for i in range(len(a_mat)):
            adjacencies = []
            for j in range(len(a_mat[i])):
                if a_mat[i][j] == 1:
                    adjacencies.append(j)
            a_list.append(adjacencies)
        return a_list

    @staticmethod
    def a_mat2e_list(a_mat: list[list[int]]) -> list[Tuple[int, int]]:
        e_list = []
        for i in range(len(a_mat)):
            for j in range(len(a_mat[i])):
                if a_mat[i][j] == 1:
                    e_list.append((i, j))
        return e_list


class A_mat_graph(Graph):

    def __init__(self, factor: int) -> None:
        super().__init__(factor)
        self.a_matrix: list[list[int]] = []

    def init_edges(self, n: int = None, user=False) -> None:
        if not user:
            self.a_matrix = self._generate(n = n, factor = self.factor)
            return
        if a_mat := self._input_graph():
            self.a_matrix = a_mat
    
    # def plot(self) -> None:
    #     mat = np.array(self.a_matrix)
    #     G = nx.DiGraph(mat)
    #     pos = nx.circular_layout(G)
    #     nx.draw(G, pos = pos, with_labels=True)
    #     plt.show()

    def _dfs(self, start: int, visited: list) -> None:
        print(start, end = " ") # process node
        for j in range(len(visited)): # find al successors of current node
            if self.a_matrix[start][j] and not visited[j]: # go deeper if not visited 
                visited[j] = 1
                self._dfs(j, visited)

    def dfs(self) -> None:
        visited = [0] * len(self.a_matrix)

        print("Graph DFS: ", end = "")
        for node, state in enumerate(visited): # run dfs for all not visited nodes
            if not state:
                self._dfs(node, visited) 

    def _bfs(self, start: int, visited: list):
        queue = [start] # init queue with starting node
        visited[start] = 1
        while queue: # till nodes in queue
            v = queue.pop(0) # take next from queue
            print(v, end = " ") # process node
            for j in range(len(visited)): # add all not visited successors of current node to queue
                if self.a_matrix[v][j] == 1 and not visited[j]:
                    queue.append(j)
                    visited[j] = 1

    def bfs(self) -> None:
        visited = [0] * len(self.a_matrix)

        print("Graph BFS: ", end = "")
        for node, state in enumerate(visited): # run bfs for all not visited nodes
            if not state:
                self._bfs(node, visited) 

    def _depth_sort(self, start: int, topo_order: list, states: list) -> None:
        if states[start] == "g": #check for cycles
            return False

        if states[start] == 'b': #check if proccessed earlier
            return True
        
        states[start] = 'g' # mark current as started

        for neighbor in range(len(self.a_matrix)): # for all successors of current node run util
            if self.a_matrix[start][neighbor]:
                if not self._depth_sort(neighbor, topo_order, states): 
                    return False
        states[start] = 'b'
        topo_order.append(start)
        return True

        

    def depth_sort(self) -> None:
        topo_order = [] # init output stack
        states = ['w'] * len(self.a_matrix) # set all nodes white
        for node, state in enumerate(states): # run util func for all white nodes
            if state == 'w': 
                if not self._depth_sort(node, topo_order, states): 
                    print("Graph is cyclic")
                    return
        print(f"topological order: {topo_order[::-1]}")


    def breadth_sort(self) -> None:
        in_deg = [0] * len(self.a_matrix) # init array of in degrees of nodes

        for i, row in enumerate(self.a_matrix):  
            for j, value in enumerate(row):
                in_deg[j] += self.a_matrix[i][j] # calc in degrees of nodes

        queue = [i for i, j in enumerate(in_deg) if j == 0] # init queue with all nodes with no predecessors
        topo_order = [] # init output stack

        while queue: # till nodes in queue
            node = queue.pop(0) # take next from queue
            topo_order.append(node) # add node to result

            for neighbor in range(len(self.a_matrix)): # update in degrees of current node successors
                if self.a_matrix[node][neighbor]:
                    in_deg[neighbor] -= 1
                    if in_deg[neighbor] == 0: # if node has no predeccessors add node to queue
                        queue.append(neighbor)
        temp = topo_order if len(topo_order) == len(self.a_matrix) else "graph is cyclic"
        print(f"topological order: {temp}")
             


    def show(self) -> None:
        print("adjacency matrix: ")
        for row in self.a_matrix:
            print(row)
        print("\nadjacency list: ")
        for i, row in enumerate(self.a_mat2a_list(self.a_matrix)):
            print(f"{i}: {row}")
        print("\nedges list: ")
        for row in self.a_mat2e_list(self.a_matrix):
            print(row)



class A_list_graph(Graph):

    def __init__(self, factor: int) -> None:
        super().__init__(factor)
        self.a_list: list[int] = []

    def init_edges(self, n: int, user=False) -> None:
        if not user:
            self.a_list = self._generate(n, self.factor)
            return
        if a_mat := self._input_graph():
            self.a_list = self.a_mat2a_list(a_mat) 

    def dfs(self) -> None:
        pass

    def bfs(self) -> None:
        pass

    def depth_sort(self) -> None:
        pass

    def breadth_sort(self) -> None:
        pass


class E_list_graph(Graph):

    def __init__(self, factor: int) -> None:
        super().__init__(factor)
        self.e_list: list[list[int]] = []

    def init_edges(self, n: int, user=False) -> None:
        if not user:
            self.e_list = self._generate(n, self.factor)
            return
        if a_mat := self._input_graph():
            self.e_list = self.a_mat2e_list(a_mat)

    def dfs(self) -> None:
        pass

    def bfs(self) -> None:
        pass

    def depth_sort(self) -> None:
        pass

    def breadth_sort(self) -> None:
        pass



