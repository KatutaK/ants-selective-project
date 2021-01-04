#!/usr/bin/env python
from random import shuffle
from ants import *
import logging

class MyBot:
    def do_turn(self, ants):
        destinations = []
        #destinations = set()

        #for coord in ants.my_ants():
        #    a_row, a_col = ants.bfs_closest_food(coord)
        #    # try all directions randomly until one is passable and not occupied
        #    directions = AIM.keys()
        #    shuffle(directions)
        #    for direction in directions:
        #        (n_row, n_col) = ants.destination(a_row, a_col, direction)
        #        if (not (n_row, n_col) in destinations and
        #                ants.passable(n_row, n_col)):
        #            ants.issue_order((a_row, a_col, direction))
        #            destinations.append((n_row, n_col))
        #            break
        #    else:
        #        destinations.append((a_row, a_col))
              
        #logging.basicConfig(file='MyBot.log',filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        for a_row, a_col in ants.my_ants():
            coord = a_row, a_col
            #logging.warning('coordinates:=>')
            #logging.warning(coord)
            #logging.warning(len(ants.my_ants()))
            # try all directions randomly until one is passable and not occupied
            #print >> open('file2.txt', 'a'), '##Ant coordinates##', 'Number of Ants: ', len(ants.my_ants())
            #print >> open('file2.txt', 'a'), 'coord', coord
            destinations.append(coord)
                
            #print >> open('file2.txt', 'a'), 'visited', destinations


            directions = AIM.keys()
            shuffle(directions)
            for direction in directions:
                #print >> open('file2.txt', 'a'), 'visited', len(destinations)
                #print >> open('file2.txt', 'a'), '##Ant possible direction coordinates##'
           
                (n_row, n_col) = ants.destination(a_row, a_col, direction)
                dest= n_row, n_col
                origin = a_row,a_col,direction

                #print >> open('file2.txt', 'a'), 'origin:',coord,'destination:',dest

                if (not (n_row, n_col) in destinations and
                        ants.passable(n_row, n_col)):
                    ants.issue_order((a_row, a_col, direction))
                    print >> open('file2.txt', 'a'), 'origin:',origin,'destination:',dest,not (n_row, n_col) in destinations
           
                    destinations.append((n_row, n_col))
                    #print >> open('file2.txt', 'a'), 'destination:',destinations
           
                    break
            else:
                destinations.append((a_row, a_col))
                

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
