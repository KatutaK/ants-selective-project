#!/usr/bin/env python
from random import randrange, choice, shuffle, randint, seed, random
from math import sqrt
from collections import deque, defaultdict
from ants import *
import logging

class MyBot:
    def __init__(self):
        self.width = None
        self.height = None
        self.map = None
        self.ant_list = {}
        self.food_list = []
        self.destinations = []
        self.FOOD =-3
        self.ants_keep_left = {}

    def do_turn(self, ants):
        #number of ants
        self.ant_list = ants.my_ants()
        #load the map
        self.map = ants.map
        #print >> open('MyBotv1_log.txt', 'a'), 'map',self.map # check that map contains values
        visited = set()
        for a_row, a_col in ants.my_ants():
            print >> open('ant_moves_log.txt', 'a'), 'Inside ants loop:' 
                    
            #ant current location
            coord = a_row, a_col
            #directions = AIM.keys()
            #shuffle(directions)
            #obtain the target destination
            target_destination = ants.closest_food(a_row, a_col)
            n_row, n_col = target_destination
            #print >> open('MyBotv1_log.txt', 'a'), 'origin',coord,'target',target_destination
            #direct ant to the food
            location_directions = ants.direction(a_row, a_col, n_row, n_col)
            #print >> open('ant_moves_log.txt', 'a'), 'origin',coord,'target',target_destination, 'directions',location_directions
            shuffle(location_directions)
            for direction in location_directions:
                #print >> open('ant_moves_log.txt', 'a'), 'source:',coord,'direction:',direction,'target:',target_destination    
                if ( ants.passable(n_row, n_col)):
                    print >> open('ant_moves_log.txt', 'a'), 'Inside directions loop:' 
                    new_ant_loc = ants.issue_order((a_row, a_col, direction))
                    print >> open('ant_moves_log.txt', 'a'), 'source:',coord,'direction:',direction,'target:',target_destination,'new_ant_loc', new_ant_loc 
                    
                else:
                    print >> open('ant_moves_log.txt', 'a'), 'not_passable:',coord,'direction:',direction,'target:',target_destination,'new_ant_loc', new_ant_loc 
                    
                    #TODO: navigate obtacle
                    self.ants_keep_left[(a_row, a_col)] = LEFT[direction]
            #TODO: navigate obstacle
            if (a_row, a_col) in self.ants_keep_left:
                direction = self.ants_keep_left[(a_row, a_col)]
                directions = [LEFT[direction], direction, RIGHT[direction], BEHIND[direction]]
                for new_direction in directions:
                    n_row, n_col = ants.destination(a_row, a_col, 'w')
                    if ants.passable(n_row, n_col):
                        ants.issue_order((a_row, a_col, new_direction))
                        break

    #nearest food sources using BFS
    def nearest_food_sources(self,start_loc,ants): 
        #return 'nearest_food_sources'
        if len(self.map) > 0 and self.map[start_loc[0]][start_loc[1]] == self.FOOD:
            return start_loc

        explored = set()
        food_queue = deque([start_loc])
        #print >> open('MyBotv1_log.txt', 'a'), 'food_queue',food_queue
        while food_queue:
            row, col = food_queue.popleft()
            c_loc = row, col
            #print >> open('MyBotv1_log.txt', 'a'), 'c_loc',c_loc
            directions = AIM.keys()
            for d in directions:
                #print >> open('MyBotv1_log.txt', 'a'), 'AIM.values()',d

                n_loc = ants.destination(row,col,d)
                #print >> open('MyBotv1_log.txt', 'a'), 'c_loc',c_loc,'n_loc',n_loc

                if n_loc in explored: continue

                if self.map[n_loc[0]][n_loc[1]] == self.FOOD:
                    return n_loc

                explored.add(n_loc)
                food_queue.append(n_loc)

        return None           

if __name__ == '__main__':
    try:
        import psyco
        psyco.full()
    except ImportError:
        pass
    try:
        Ants.run(MyBot())
    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
