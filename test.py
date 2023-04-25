from models import *
import pandas as pd
import time
from collections import defaultdict
import sys
import threading

sys.setrecursionlimit(10000)
threading.stack_size(0x2000000)



FILE = 'results.txt'
DSIZE = [100 * i for i in range(1, 11)]
TRIES = 2
FACTOR = .5


def _benchmark(model, overdata):

    Frame = pd.DataFrame()
    for index, i in enumerate(overdata):
        graph = model(FACTOR)
        graph.init_edges(i)


        for q in range(20):
            frame = defaultdict(dict)
            for func in [graph.bfs, graph.dfs, graph.depth_sort, graph.breadth_sort]:
                start = time.time()
                func()
                frame[index][func.__name__] = time.time() - start
            frame[index]['dsize'] = i
            frame[index]['type'] = model.__name__
            frame = pd.DataFrame.from_dict(frame, orient='index')
            Frame = pd.concat([Frame, frame])
        print(f"Done {i}")
    print(Frame.head())
    
    return Frame


def main():

    frame = pd.DataFrame()

    for i in range(TRIES):
        frame = pd.concat([_benchmark(E_list_graph, DSIZE), pd.DataFrame(),
                           frame],
                          ignore_index=True)


    frame.to_csv(FILE)


t = threading.Thread(target=main())
t.start()
t.join()