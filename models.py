from typing import Tuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import itertools


class Graph:

    def __init__(self, factor: float) -> None:
        self.factor = factor

    @staticmethod
    def _generate(n: int, factor: float) -> list[list[int]]:
        possible_edges = n * (n - 1) / 2
        edges = list(itertools.chain.from_iterable([[(i, j) for i in range(n) if i < j] for j in range(n)]))
        random.shuffle(edges)
        ind = 0
        fill_edges = math.floor(factor * possible_edges) - n
        a_mat = [[0 for _ in range(n)] for _ in range(n)]
        # make sure graph will be connected
        nodes = [i for i in range(n)]
        np.random.shuffle(nodes)
        last = nodes[0]
        for node in nodes[1:]:
            from_, to = min(last, node), max(last, node)
            a_mat[from_][to] = 1

        # fill rest of edges
        for i in range(fill_edges):
            while 1:
                (from_, to) = edges[ind]
                ind += 1
                if a_mat[from_][to] != 1:
                    a_mat[from_][to] = 1
                    break

        return a_mat

    @staticmethod
    def _input_graph() -> bool | list[list[int]]:
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
    def a_mat2a_list(a_mat: list[list[int]]) -> list[list[int]]:
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

    def __init__(self, factor: float) -> None:
        super().__init__(factor)
        self.a_matrix: list[list[int]] = []

    def init_edges(self, n: int = None, user=False) -> None:
        if not user:
            self.a_matrix = self._generate(n=n, factor=self.factor)
            return
        if a_mat := self._input_graph():
            self.a_matrix = a_mat

    def plot(self) -> None:
        mat = np.array(self.a_matrix)
        G = nx.DiGraph(mat)
        pos = nx.circular_layout(G)
        nx.draw(G, pos=pos, with_labels=True)
        plt.show()

    def _dfs(self, start: int, visited: list) -> None:
        print(start, end=" ")  # process node
        for j in range(len(visited)):  # find al successors of current node
            if self.a_matrix[start][j] and not visited[j]:  # go deeper if not visited
                visited[j] = 1
                self._dfs(j, visited)

    def dfs(self) -> None:
        visited = [0] * len(self.a_matrix)

        print("Graph DFS: ", end="")
        for node, state in enumerate(visited):  # run dfs for all not visited nodes
            if not state:
                self._dfs(node, visited)

    def _bfs(self, start: int, visited: list):
        queue = [start]  # init queue with starting node
        visited[start] = 1
        while queue:  # till nodes in queue
            v = queue.pop(0)  # take next from queue
            print(v, end=" ")  # process node
            for j in range(len(visited)):  # add all not visited successors of current node to queue
                if self.a_matrix[v][j] == 1 and not visited[j]:
                    queue.append(j)
                    visited[j] = 1

    def bfs(self) -> None:
        visited = [0] * len(self.a_matrix)

        print("Graph BFS: ", end="")
        for node, state in enumerate(visited):  # run bfs for all not visited nodes
            if not state:
                self._bfs(node, visited)

    def _depth_sort(self, start: int, topo_order: list, states: list) -> None | bool:
        if states[start] == "g":  # check for cycles
            return False

        if states[start] == 'b':  # check if processed earlier
            return True

        states[start] = 'g'  # mark current as started

        for neighbor in range(len(self.a_matrix)):  # for all successors of current node run util
            if self.a_matrix[start][neighbor]:
                if not self._depth_sort(neighbor, topo_order, states):
                    return False
        states[start] = 'b'
        topo_order.append(start)
        return True

    def depth_sort(self) -> None:
        topo_order = []  # init output stack
        states = ['w'] * len(self.a_matrix)  # set all nodes white
        for node, state in enumerate(states):  # run util func for all white nodes
            if state == 'w':
                if not self._depth_sort(node, topo_order, states):
                    print("Graph is cyclic")
                    return
        print(f"topological order: {topo_order[::-1]}")

    def breadth_sort(self) -> None:
        in_deg = [0] * len(self.a_matrix)  # init array of in degrees of nodes

        for i, row in enumerate(self.a_matrix):
            for j, value in enumerate(row):
                in_deg[j] += self.a_matrix[i][j]  # calc in degrees of nodes

        queue = [i for i, j in enumerate(in_deg) if j == 0]  # init queue with all nodes with no predecessors
        topo_order = []  # init output stack

        while queue:  # till nodes in queue
            node = queue.pop(0)  # take next from queue
            topo_order.append(node)  # add node to result

            for neighbor in range(len(self.a_matrix)):  # update in degrees of current node successors
                if self.a_matrix[node][neighbor]:
                    in_deg[neighbor] -= 1
                    if in_deg[neighbor] == 0:  # if node has no predecessors add node to queue
                        queue.append(neighbor)
                        in_deg[neighbor] -= 1
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

    def __init__(self, factor: float) -> None:
        super().__init__(factor)
        self.a_list: list[list[int]] = []

    def init_edges(self, n: int, user=False) -> None:
        if not user:
            self.a_list = self.a_mat2a_list(self._generate(n, self.factor))
            return
        if a_mat := self._input_graph():
            self.a_list = self.a_mat2a_list(a_mat)

    def _dfs(self, start: int, visited: list) -> None:
        print(start, end=" ")  # process node
        for j in self.a_list[start]:  # find al successors of current node
            if not visited[j]:  # go deeper if not visited
                visited[j] = 1
                self._dfs(j, visited)

    def dfs(self) -> None:
        visited = [0] * len(self.a_list)

        print("Graph DFS: ", end="")
        for node, state in enumerate(visited):  # run dfs for all not visited nodes
            if not state:
                self._dfs(node, visited)

    def _bfs(self, start: int, visited: list):
        queue = [start]  # init queue with starting node
        visited[start] = 1
        while queue:  # till nodes in queue
            v = queue.pop(0)  # take next from queue
            print(v, end=" ")  # process node
            for j in self.a_list[start]:  # add all not visited successors of current node to queue
                if not visited[j]:
                    queue.append(j)
                    visited[j] = 1

    def bfs(self) -> None:
        visited = [0] * len(self.a_list)

        print("Graph BFS: ", end="")
        for node, state in enumerate(visited):  # run bfs for all not visited nodes
            if not state:
                self._bfs(node, visited)

    def _depth_sort(self, start: int, topo_order: list, states: list) -> None | bool:
        if states[start] == "g":  # check for cycles
            return False

        if states[start] == 'b':  # check if processed earlier
            return True

        states[start] = 'g'  # mark current as started

        for neighbor in self.a_list[start]:  # for all successors of current node run util
            if not self._depth_sort(neighbor, topo_order, states):
                return False

        states[start] = 'b'
        topo_order.append(start)
        return True

    def depth_sort(self) -> None:
        topo_order = []  # init output stack
        states = ['w'] * len(self.a_list)  # set all nodes white
        for node, state in enumerate(states):  # run util func for all white nodes
            if state == 'w':
                if not self._depth_sort(node, topo_order, states):
                    print("Graph is cyclic")
                    return
        print(f"topological order: {topo_order[::-1]}")

    def breadth_sort(self) -> None:
        in_deg = [0] * len(self.a_list)  # init array of in degrees of nodes

        for i, row in enumerate(self.a_list):
            for neighbor in row:
                in_deg[neighbor] += 1  # calc in degrees of nodes

        queue = [i for i, j in enumerate(in_deg) if j == 0]  # init queue with all nodes with no predecessors
        topo_order = []  # init output stack

        while queue:  # till nodes in queue
            node = queue.pop(0)  # take next from queue
            topo_order.append(node)  # add node to result

            for neighbor in self.a_list[node]:  # update in degrees of current node successors
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:  # if node has no predecessors add node to queue
                    queue.append(neighbor)
                    in_deg[neighbor] -= 1

        temp = topo_order if len(topo_order) == len(self.a_list) else "graph is cyclic"
        print(f"topological order: {temp}")


class E_list_graph(Graph):

    def __init__(self, factor: float) -> None:
        super().__init__(factor)
        self.e_list: list[Tuple[int, int]] = []
        self.n: int = 0

    def init_edges(self, n: int, user=False) -> None:
        self.n = n
        if not user:
            self.e_list = self.a_mat2e_list(self._generate(n, self.factor))
            return
        if a_mat := self._input_graph():
            self.e_list = self.a_mat2e_list(a_mat)

    def _dfs(self, start: int, visited: list) -> None:
        print(start, end=" ")  # process node
        for from_, to in self.e_list:  # find al successors of current node
            if from_ == start and not visited[to]:  # go deeper if not visited
                visited[to] = 1
                self._dfs(to, visited)

    def dfs(self) -> None:
        visited = [0] * self.n

        print("Graph DFS: ", end="")
        for node, state in enumerate(visited):  # run dfs for all not visited nodes
            if not state:
                self._dfs(node, visited)

    def _bfs(self, start: int, visited: list):
        queue = [start]  # init queue with starting node
        visited[start] = 1
        while queue:  # till nodes in queue
            v = queue.pop(0)  # take next from queue
            print(v, end=" ")  # process node
            for from_, to in self.e_list:  # add all not visited successors of current node to queue
                if from_ == start and not visited[to]:
                    queue.append(to)
                    visited[to] = 1

    def bfs(self) -> None:
        visited = [0] * self.n

        print("Graph BFS: ", end="")
        for node, state in enumerate(visited):  # run bfs for all not visited nodes
            if not state:
                self._bfs(node, visited)

    def _depth_sort(self, start: int, topo_order: list, states: list) -> None | bool:
        if states[start] == "g":  # check for cycles
            return False

        if states[start] == 'b':  # check if processed earlier
            return True

        states[start] = 'g'  # mark current as started

        for from_, to in self.e_list:  # for all successors of current node run util
            if from_ == start:
                if not self._depth_sort(to, topo_order, states):
                    return False
        states[start] = 'b'
        topo_order.append(start)
        return True

    def depth_sort(self) -> None:
        topo_order = []  # init output stack
        states = ['w'] * self.n  # set all nodes white
        for node, state in enumerate(states):  # run util func for all white nodes
            if state == 'w':
                if not self._depth_sort(node, topo_order, states):
                    print("Graph is cyclic")
                    return
        print(f"topological order: {topo_order[::-1]}")

    def breadth_sort(self) -> None:
        in_deg = [0] * self.n  # init array of in degrees of nodes

        for from_, to in self.e_list:
            in_deg[to] += 1  # calc in degrees of nodes

        queue = [i for i, j in enumerate(in_deg) if j == 0]  # init queue with all nodes with no predecessors
        topo_order = []  # init output stack

        while queue:  # till nodes in queue
            node = queue.pop(0)  # take next from queue
            topo_order.append(node)  # add node to result

            for from_, to in self.e_list:  # update in degrees of current node successors
                if from_ == node:
                    in_deg[to] -= 1
                if in_deg[to] == 0:  # if node has no predecessors add node to queue
                    queue.append(to)
                    in_deg[to] -= 1

        temp = topo_order if len(topo_order) == self.n else "graph is cyclic"
        print(f"topological order: {temp}")
