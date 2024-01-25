#!/usr/bin/env python3
# solver2023.py : 2023 Sliding tile puzzle solver
#
# Code by: Niveditha Bommanahally Parameshwarappa(nibomm), Dhruvil Mansukhbhai Dholariya(ddholari), Bindu Madhavi Dokala(bdokala)
#
# Based on skeleton code by B551 Staff, Fall 2023
#

import sys
import heapq
import numpy as np

ROWS = 5
COLS = 5

def printable_board(state):
    return [ ('%3d ')*COLS  % state[j:(j+COLS)] for j in range(0, ROWS*COLS, COLS) ]


def get_manhattan_distance(state, goal_state):
    board = np.array(state)
    goal_state = np.array(goal_state)

    ROWS, COLS = board.shape

    man_dist = 0
    for i in range(ROWS):
        for j in range(COLS):
            num = board[i, j]
            if num != 0:  # Skip the empty tile (if applicable)
                goal_pos = np.argwhere(goal_state == num)[0]
                man_dist += np.abs(np.array([i, j]) - goal_pos).sum()
    #added scaling factors 0.2 as 1 was not giving optimal solutions (tried 0.3 and 0.5 still 0.2 was good), I reduced the weightage of this heuristics
    return man_dist * 0.2

def is_goal(state):
    goal_state = np.arange(1, 26).reshape(ROWS, COLS)
    return np.all(state == goal_state)


def apply_move(state, move_list, move):
    new_state = np.copy(state)

    # Define the indices for the inner circle
    inner_circle_indices = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1)]
    # Define the indices for the outer circle
    outer_circle_indices = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (4, 3), (4, 2), (4, 1), (4, 0), (3, 0), (2, 0), (1, 0)]

    if move[0] == 'U':
        index = int(move[1])-1
        #print(f"move , idex ",move,index)
        new_state[:, index] = np.roll(new_state[:, index], -1, axis=0)

    elif move[0] == 'D':
        index = int(move[1])-1
        #print(f"move , idex ",move,index)
        new_state[:, index] = np.roll(new_state[:, index], 1, axis=0)

    elif move[0] == 'L':
        index = int(move[1])-1
        #print(f"move , idex ",move,index)
        new_state[index, :] = np.roll(new_state[index, :], -1, axis=0)

    elif move[0] == 'R':
        index = int(move[1])-1
        #print(f"move , idex ",move,index)
        new_state[index, :] = np.roll(new_state[index, :], 1, axis=0)
    # As matrix always 5X5 hard coded the inner and outer rotation for efficency , not the coding standard i understand
    elif move == 'Occ':
        # new_state[0][0]	=	state[1][0]
        # new_state[0][1]	=	state[0][0]
        # new_state[0][2]	=	state[0][1]
        # new_state[0][3]	=	state[0][2]
        # new_state[0][4]	=	state[0][3]
        # new_state[1][4]	=	state[0][4]
        # new_state[2][4]	=	state[1][4]
        # new_state[3][4]	=	state[2][4]
        # new_state[4][4]	=	state[3][4]
        # new_state[4][3]	=	state[4][4]
        # new_state[4][2]	=	state[4][3]
        # new_state[4][1]	=	state[4][2]
        # new_state[4][0]	=	state[4][1]
        # new_state[3][0]	=	state[4][0]
        # new_state[2][0]	=	state[3][0]
        # new_state[1][0]	=	state[2][0]
        # Apply the 'Oc' operation to the outer circle
        for i in range(len(outer_circle_indices)):
            row, col = outer_circle_indices[i]
            new_state[row][col] = state[outer_circle_indices[(i + 1) % len(outer_circle_indices)][0]][outer_circle_indices[(i + 1) % len(outer_circle_indices)][1]]


    elif move == 'Oc':
        # new_state[0][0]	=	state[0][1]
        # new_state[0][1]	=	state[0][2]
        # new_state[0][2]	=	state[0][3]
        # new_state[0][3]	=	state[0][4]
        # new_state[0][4]	=	state[1][4]
        # new_state[1][4]	=	state[2][4]
        # new_state[2][4]	=	state[3][4]
        # new_state[3][4]	=	state[4][4]
        # new_state[4][4]	=	state[4][3]
        # new_state[4][3]	=	state[4][2]
        # new_state[4][2]	=	state[4][1]
        # new_state[4][1]	=	state[4][0]
        # new_state[4][0]	=	state[3][0]
        # new_state[3][0]	=	state[2][0]
        # new_state[2][0]	=	state[1][0]
        # new_state[1][0]	=	state[0][0]
        for i in range(len(outer_circle_indices)):
            row, col = outer_circle_indices[i]
            new_state[row][col] = state[outer_circle_indices[(i - 1) % len(outer_circle_indices)][0]][outer_circle_indices[(i - 1) % len(outer_circle_indices)][1]]

    elif move == 'Ic':
        # new_state[1][1]	=	state[2][1]
        # new_state[1][2]	=	state[1][1]
        # new_state[1][3]	=	state[1][2]
        # new_state[2][3]	=	state[1][3]
        # new_state[3][3]	=	state[2][3]
        # new_state[3][2]	=	state[3][3]
        # new_state[3][1]	=	state[3][2]
        # new_state[2][1]	=	state[3][1]
        for i in range(len(inner_circle_indices)):
            row, col = inner_circle_indices[i]
            new_state[row][col] = state[inner_circle_indices[(i - 1) % len(inner_circle_indices)][0]][inner_circle_indices[(i - 1) % len(inner_circle_indices)][1]]


    elif move == 'Icc':
        # new_state[1][1]	=	state[1][2]
        # new_state[1][2]	=	state[1][3]
        # new_state[1][3]	=	state[2][3]
        # new_state[2][3]	=	state[3][3]
        # new_state[3][3]	=	state[3][2]
        # new_state[3][2]	=	state[3][1]
        # new_state[3][1]	=	state[2][1]
        # new_state[2][1]	=	state[1][1]
        for i in range(len(inner_circle_indices)):
            row, col = inner_circle_indices[i]
            new_state[row][col] = state[inner_circle_indices[(i + 1) % len(inner_circle_indices)][0]][inner_circle_indices[(i + 1) % len(inner_circle_indices)][1]]

    return new_state , move_list+[move]

# def print_board(board):
#     flat_board = tuple([item for sublist in board for item in sublist])
#     tmp = [ ('%3d ')*COLS  % flat_board[j:(j+COLS)] for j in range(0, ROWS*COLS, COLS) ]
#     print("\n".join(tmp))

def successors(state, move_list):
    successors = []
    moves_set = ["L5","L1","L2","L3","L4",
                 "R5","R1","R2","R3","R4",
                 "U5","U1","U2","U3","U4",
                 "D5","D1","D2","D3","D4",
                 "Oc","Occ","Ic","Icc"]
    for move in moves_set:
        successors.append((apply_move(state , move_list, move)))
    return successors

def solve(initial_board):
    """
    1. This function should return the solution as instructed in the assignment, consisting of a list of moves like ["R2","D2","U1"].
    2. Do not add any extra parameters to the solve() function, or it will break our grading and testing code.
       For testing, we will call this function with a single argument (initial_board), and it should return 
       the solution.
    3. Please do not use any global variables, as it may cause the testing code to fail.
    4. You can assume that all test cases will be solvable.
    5. The current code just returns a dummy solution.
    """
    # Reshape the flat representation into a 5x5 matrix

# Define your custom data structures or variables here
    fringe = []
    closed_states = set()
    #numpy and heapq for efficiency to hand arrays
    goal_state = np.arange(1, 26).reshape(ROWS, COLS)
    initial_board = np.array(initial_board).reshape(ROWS, COLS)

    # Calculate the initial priority based on the heuristic f(x) = g(x) + h(x)
    # g(x) cost function length of moves taken
    # h(x) heuristic function is manhattan distance between board and goal state
    initial_priority = len([]) + get_manhattan_distance(initial_board, goal_state)
    heapq.heappush(fringe, (initial_priority, (tuple(map(tuple, initial_board)), [])))

    while fringe:
        _, (current_state, move_list) = heapq.heappop(fringe)

        if is_goal(current_state):
            return move_list

        if tuple(map(tuple, current_state)) in closed_states:
            continue

        closed_states.add(tuple(map(tuple, current_state)))

        for board, move_hist in successors(current_state, move_list):
            # Calculate the priority for the next state
            next_priority = len(move_hist) + get_manhattan_distance(board, goal_state)
            heapq.heappush(fringe, (next_priority, (tuple(map(tuple, board)), move_hist)))

    # Return the solution path as a list of moves

# Please don't modify anything below this line
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Error: expected a board filename")

    start_state = []
    with open(sys.argv[1], 'r') as file:
        for line in file:
            start_state += [int(i) for i in line.split()]

    if len(start_state) != ROWS * COLS:
        raise Exception("Error: couldn't parse the start state file")

    print("Start state: \n" + "\n".join(printable_board(tuple(start_state))))

    print("Solving...")
    route = solve(tuple(start_state))

    print("Solution found in " + str(len(route)) + " moves:" + "\n" + " ".join(route))
    matrix = np.array(start_state).reshape(ROWS, COLS)
    # while True:
    #     input_str = input("Enter a move and index (e.g., 'Oc L0', 's' for restart, 'q' for quit): ")
    #     if input_str == 'q':
    #         break  # Quit the program
    #     elif input_str == 's':
    #         matrix = matrix  # Restart from the first state
    #         print("Restarted to the first state.")
    #         print_board(matrix)
    #     else:
    #             matrix = apply_move(matrix, input_str)
    #             print("Current state: ")
    #             print_board(matrix)
    #             print(is_goal(matrix))