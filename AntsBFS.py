#!/usr/bin/env python
import sys
from search import *

#subclass the Problem class
class BFSAntsProblem(Problem):
    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. """
        super().__init__(initial, goal)

# Gather our code in a main() function
def main():
    g =  Graph({'A': {'B': 1, 'C': 2}})
    print(g.nodes())
    print ("Hello there")
    # Command line args are in sys.argv[1], sys.argv[2] ...
    # sys.argv[0] is the script name itself and can be ignored

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()