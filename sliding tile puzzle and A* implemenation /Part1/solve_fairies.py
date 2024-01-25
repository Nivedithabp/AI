#!/usr/local/bin/python3
# solve_fairies.py : Fairy puzzle solver
#
# Code by: Niveditha Bommanahally Parameshwarappa(nibomm), Dhruvil Mansukhbhai Dholariya(ddholari), Bindu Madhavi Dokala(bdokala)
#
# Based on skeleton code by B551 course staff, Fall 2023
#
# N fairies stand in a row on a wire, each adorned with a magical symbol from 1 to N.
# In a single step, two adjacent fairies can swap places. How can
# they rearrange themselves to be in order from 1 to N in the fewest
# possible steps?

# !/usr/bin/env python3
import sys
import heapq

N=5

#####
# THE ABSTRACTION:
#
# Initial state:

# Goal state:
# given a state, returns True or False to indicate if it is the goal state
def is_goal(state):
    return state == list(range(1, N+1))

# Successor function:
# given a state, return a list of successor states
def successors(state):
    return [ state[0:n] + [state[n+1],] + [state[n],] + state[n+2:] for n in range(0, N-1) ]

# Heuristic function:
# given a state, return an estimate of the number of steps to a goal from that state
# The admissible heuristic return the next possible state of the fairies
def h(pres_state):
    # taking the goal state as 12345
    goal_state = [1,2,3,4,5] 
    res = 0
    # checking by how many positions each fairy has misplaced 
    # adding the misplaced positions sum of each fairy to return the heuristic funcation value
    for i in range(5):
         res+=abs(goal_state[i]-pres_state[i])
    return res

#########
#
# THE ALGORITHM:
#
# This is a generic solver using BFS. 
#
def solve(start_state):
    # taking a Priority queue to implement A* using a admissible heuristic
    fringe = []  
   
    # Taking  initial cost as 0 as cost from the initial state of fairy to itself is 0.
    initial_g_cost = 0 
    # Pushing the f_cost, initial cost, intial state to heapq
    heapq.heappush(fringe, (initial_g_cost + h(start_state), initial_g_cost, start_state, []))
    # till the heapq is not empty iterate over the heapq
    while fringe:
        # pop the heapq to get the current state with the lowest f_cost
        f_c, present_g_cost, present_state, present_path = heapq.heappop(fringe)  
        # if current state is goal state return path
        if is_goal(present_state):
            return present_path + [present_state]
        # if goal state is not yet reached, find neighbors of the fairy and iterate over them
        for fairy_neighbor in successors(present_state):
            # Finding g_cost for each successor the fairy. for each swap we make, it costs 1 
            g_cost_successor = present_g_cost + 1  

            # Finding the f_cost of the A* by adding the computed g_cost and heuristic of successor
            f_cost_successor = g_cost_successor + h(fairy_neighbor)
            # Pushing the f_cost, g_cost, successor of fairy, and the path to heapq.
            heapq.heappush(fringe, (f_cost_successor, g_cost_successor, fairy_neighbor, present_path + [present_state]))

    return []


# Please don't modify anything below this line
#
if __name__ == "__main__":
    if(len(sys.argv) != 2):
        raise(Exception("Error: expected a test case filename"))

    test_cases = []
    with open(sys.argv[1], 'r') as file:
        for line in file:
            test_cases.append([ int(i) for i in line.split() ])
    for initial_state in test_cases:
        	print('From state ' + str(initial_state) + " found goal state by taking path: " + str(solve(initial_state)))

    

