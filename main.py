import sys
import threading
from models import *
import os


sys.setrecursionlimit(10000)
threading.stack_size(0x2000000)




def clear() -> None:
    os.system('cls')


# At end of line [possible commands]: indicates waiting for user action
def parser(command: str) -> None:
    command = command.split(',')
    command = [i.lower() for i in command]
    command = [int(i) if i.isdigit() else i.replace(" ", "").replace("-", '') for i in command]

    for instruction in command:
        if instruction in ['show']:
            graph.show()
            input("[any]: ")

        elif instruction in ["dfs"]:
            graph.dfs()
            input("[any]: ")
        
        elif instruction in ["bfs"]:
            graph.bfs()
            input("[any]: ")

        elif instruction in ["dsort"]:
            graph.depth_sort()
            input("[any]: ")

        elif instruction in ["bsort"]:
            graph.breadth_sort()
            input("[any]: ")

        # elif instruction in ['plot']:
        #     graph.plot()


        elif instruction in ['h', 'help']:
            print("dfs")
            print("bfs")
            print("dsort - topological sort depth method")
            print("bsort - topological sort breadth method")
            print("show - print graph representations")
            input("[any]: ")
        else:
            input("Command invalid [any]: ")


graph = A_mat_graph(.5)

def main() -> None:
    global graph

    menu: str = 'default'
    tree_type: str = ''
    array: list = []

    while True:

        if menu == "default":
            clear()
            input("Graph commander [any]: ")
            menu = 'choose data'

        elif menu == "choose data":
            fail = False
            while 1:
                clear()
                if fail:
                    print('I don\'t understand. use [enter/gen/exit]')
                temp = input("Enter data or generate [enter/gen/exit]: ").lower()
                if temp in ['gen', 'enter', 'exit']:
                    break
                fail = True
            if temp == 'exit':
                break

            menu = temp

        elif menu == 'gen':
            fail = False
            while 1:
                clear()
                if fail:
                    print('I don\'t understand. use [n: int > 0]')
                temp = input("Enter number of nodes [n: int > 0/exit]: ").lower()
                if temp == 'exit' or (temp.isdigit() and int(temp) > 0):
                    break
                fail = True
            if temp == 'exit':
                break
            n = int(temp)
            graph.init_edges(n, user = False)
            menu = 'choose func'

        elif menu == 'enter':
            graph.init_edges(user = True)
            input("[any]: ")
            menu = 'choose func'


        elif menu == "choose func":
            clear()

            command = input("Enter command (h for help) [valid command/h/exit]: ")
            if command == 'exit':
                break
            parser(command)
    clear()




t = threading.Thread(target = main())
t.start()
t.join()

