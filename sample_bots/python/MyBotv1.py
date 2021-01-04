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
        self.ants =1
    def do_turn(self, ants):
        #destinations = set()

        self.ants = len(ants.my_ants())
        for a_row, a_col in ants.my_ants():
            #clear destinations to enable new ants move
            if self.ants > 1:
                self.destinations = []
            coord = a_row, a_col
            directions = AIM.keys()
            shuffle(directions)
            for direction in directions:
                (n_row, n_col) = ants.destination(a_row, a_col, direction)
                dest= n_row, n_col
                origin = a_row,a_col,direction
                if (not (n_row, n_col,direction) in self.destinations and
                        ants.passable(n_row, n_col)):
                    ants.issue_order((a_row, a_col, direction))
                    print >> open('MyBotv1_log.txt', 'a'), 'origin:',origin,'destination:',dest,self.destinations
                    self.destinations.append((a_row, a_col,direction))
                    break
                    #print >> open('file2.txt', 'a'), 'destination:',destinations
                

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
