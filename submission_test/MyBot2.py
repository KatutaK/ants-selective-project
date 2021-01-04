#!/usr/bin/env python
from ants import *
from search import *


FOOD = -3
# define a class with a do_turn method
# the Ants.run method will parse and update bot input
# it will also run the do_turn method for us
class MyBot:
    def __init__(self):
        # define class level variables, will be remembered between turns
        pass
    
    # do_setup is run once at the start of the game
    # after the bot has received the game settings
    # the ants class is created and setup by the Ants.run method
    def do_setup(self, ants):
        # initialize data structures after learning the game settings
        pass
    
    # do turn is run once per turn
    # the ants class has the game state and is updated by the Ants.run method
    # it also has several helper methods to use
    def do_turn(self, ants):
        # loop through all my ants and try to give them orders
        # the ant_loc is an ant location tuple in (row, col) form
        #pass the problem to the algorithm to resolve using breadth_first
        #breadth_first_graph_search(problem)
        for ant_loc in ants.my_ants():
            
            # try all directions in given order
            #directions = ('n','e','s','w')
            #directions = ('s','w','e','n')
            #for direction in directions:
                # the destination method will wrap around the map properly
                # and give us a new (row, col) tuple
                #new_loc = ants.destination(ant_loc,direction)
                # passable returns true if the location is land
                #ants.issue_order((ant_loc, direction))
                #if (ants.passable(new_loc)):
                    # an order is the location of a current ant and a direction
                    #ants.issue_order((ant_loc, direction))
                    # stop now, don't give 1 ant multiple orders
                    #break
            # check if we still have time left to calculate more orders
            #if ants.time_remaining() < 10:
                #break
    
    def find_closest_food(self, coord):
        """ Find the closest square to coord which is a foo square using BFS

            Return None if no food is found
        """
        if self.map[coord[0]][coord[1]] == FOOD:
            return coord

        visited = set()
        square_queue = deque([coord])

        while square_queue:
            c_loc = square_queue.popleft()

            for d in AIM.values():
                n_loc = self.destination(c_loc, d)
                if n_loc in visited: continue

                if self.map[n_loc[0]][n_loc[1]] == FOOD:
                      ants.issue_order((n_loc, d)) # move ant to the food
                    return n_loc

                visited.add(n_loc)
                square_queue.append(n_loc)

        return None

            
if __name__ == '__main__':
    # psyco will speed up python a little, but is not needed
    try:
        import psyco
        psyco.full()
    except ImportError:
        pass
    
    try:
        # if run is passed a class with a do_turn method, it will do the work
        # this is not needed, in which case you will need to write your own
        # parsing function and your own game state class
        Ants.run(MyBot())
    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
