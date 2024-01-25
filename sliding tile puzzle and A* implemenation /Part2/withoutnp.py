#!/usr/local/bin/python3
# solver2023.py : 2023 Sliding tile puzzle solver
#
# Code by: name IU ID
#
# Based on skeleton code by B551 Staff, Fall 2023
#

import sys
import heapq

ROWS=5
COLS=5
GOAL_STATE = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
]

SLIDE_ROW_LEFT = 'L'
SLIDE_ROW_RIGHT = 'R'
SLIDE_COL_UP = 'U'
SLIDE_COL_DOWN = 'D'
ROTATE_OUTER_CLOCKWISE = 'Oc'
ROTATE_OUTER_COUNTERCLOCKWISE = 'Occ'
ROTATE_INNER_CLOCKWISE = 'Ic'
ROTATE_INNER_COUNTERCLOCKWISE = 'Icc'

def printable_board(board):
    return [ ('%3d ')*COLS  % board[j:(j+COLS)] for j in range(0, ROWS*COLS, COLS) ]

def apply_move(state, move, index):
    new_state = [list(row) for row in state]
    if move == SLIDE_COL_UP:
        # Save the top tile value
        top_tile = new_state[0][index]

        # Move each tile in the column one position up
        for i in range(ROWS - 1):
            new_state[i][index] = new_state[i + 1][index]

        # Place the saved top tile value in the bottom row
        new_state[-1][index] = top_tile

    elif move == SLIDE_COL_DOWN:
        # Save the bottom tile value
        bottom_tile = new_state[-1][index]

        # Move each tile in the column one position down
        for i in range(ROWS - 1, 0, -1):
            new_state[i][index] = new_state[i - 1][index]

        # Place the saved bottom tile value in the top row
        new_state[0][index] = bottom_tile

    elif move == SLIDE_ROW_LEFT:
        # Save the leftmost tile value
        leftmost_tile = new_state[index][0]

        # Move each tile in the row one position to the left
        for i in range(COLS - 1):
            new_state[index][i] = new_state[index][i + 1]

        # Place the saved leftmost tile value in the last column
        new_state[index][-1] = leftmost_tile

    elif move == SLIDE_ROW_RIGHT:
        # Save the rightmost tile value
        rightmost_tile = new_state[index][-1]

        # Move each tile in the row one position to the right
        for i in range(COLS - 1, 0, -1):
            new_state[index][i] = new_state[index][i - 1]

        # Place the saved rightmost tile value in the first column
        new_state[index][0] = rightmost_tile
    elif move == ROTATE_OUTER_CLOCKWISE:
        new_state[0][0]	=	state[1][0]
        new_state[0][1]	=	state[0][0]
        new_state[0][2]	=	state[0][1]
        new_state[0][3]	=	state[0][2]
        new_state[0][4]	=	state[0][3]
        new_state[1][4]	=	state[0][4]
        new_state[2][4]	=	state[1][4]
        new_state[3][4]	=	state[2][4]
        new_state[4][4]	=	state[3][4]
        new_state[4][3]	=	state[4][4]
        new_state[4][2]	=	state[4][3]
        new_state[4][1]	=	state[4][2]
        new_state[4][0]	=	state[4][1]
        new_state[3][0]	=	state[4][0]
        new_state[2][0]	=	state[3][0]
        new_state[1][0]	=	state[2][0]

    elif move == ROTATE_OUTER_COUNTERCLOCKWISE:
        new_state[0][0]	=	state[0][1]
        new_state[0][1]	=	state[0][2]
        new_state[0][2]	=	state[0][3]
        new_state[0][3]	=	state[0][4]
        new_state[0][4]	=	state[1][4]
        new_state[1][4]	=	state[2][4]
        new_state[2][4]	=	state[3][4]
        new_state[3][4]	=	state[4][4]
        new_state[4][4]	=	state[4][3]
        new_state[4][3]	=	state[4][2]
        new_state[4][2]	=	state[4][1]
        new_state[4][1]	=	state[4][0]
        new_state[4][0]	=	state[3][0]
        new_state[3][0]	=	state[2][0]
        new_state[2][0]	=	state[1][0]
        new_state[1][0]	=	state[0][0]


    elif move == ROTATE_INNER_CLOCKWISE:
        new_state[1][1]	=	state[2][1]
        new_state[1][2]	=	state[1][1]
        new_state[1][3]	=	state[1][2]
        new_state[2][3]	=	state[1][3]
        new_state[3][3]	=	state[2][3]
        new_state[3][2]	=	state[3][3]
        new_state[3][1]	=	state[3][2]
        new_state[2][1]	=	state[3][1]
        
    elif move == ROTATE_INNER_COUNTERCLOCKWISE:
        new_state[1][1]	=	state[1][2]
        new_state[1][2]	=	state[1][3]
        new_state[1][3]	=	state[2][3]
        new_state[2][3]	=	state[3][3]
        new_state[3][3]	=	state[3][2]
        new_state[3][2]	=	state[3][1]
        new_state[3][1]	=	state[2][1]
        new_state[2][1]	=	state[1][1]

    return new_state

# return a list of possible successor states
def successors(state):
    successors = []
    # Generate successors for rotating the outer ring clockwise (Oc) and counterclockwise (Occ)
    for move in [ROTATE_OUTER_CLOCKWISE, ROTATE_OUTER_COUNTERCLOCKWISE]:
        successors.append((apply_move(state, move, None), move))

    # Generate successors for rotating the inner ring clockwise (Ic) and counterclockwise (Icc)
    for move in [ROTATE_INNER_CLOCKWISE, ROTATE_INNER_COUNTERCLOCKWISE]:
        successors.append((apply_move(state, move, None), move))

    for row in range(ROWS):
        # Generate successors for sliding rows left (L) and right (R)
        for move in [SLIDE_ROW_LEFT, SLIDE_ROW_RIGHT]:
            successors.append((apply_move(state, move, row),move + str(row+1) ))

    for col in range(COLS):
        # Generate successors for sliding columns up (U) and down (D)
        for move in [SLIDE_COL_UP, SLIDE_COL_DOWN]:
            successors.append((apply_move(state, move, col), move + str(col+1)))

    
    #print(successors)
    return successors

# check if we've reached the goal
def is_goal(state):
    return state == GOAL_STATE
def h(state):
    # Calculate the Manhattan distance from the current state to the goal state
    distance = 0

    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] != GOAL_STATE[i][j]:
                goal_value = state[i][j]

                # Find the goal position in the goal state
                goal_row, goal_col = find_position(GOAL_STATE, goal_value)
                distance += abs(i - goal_row) + abs(j - goal_col)

    return distance * 0.2



# Function to find the position (row, col) of a value in a matrix
def find_position(matrix, value):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == value:
                return i, j
    return -1, -1  # Return -1, -1 if the value is not found in the matrix


def solve(initial_board):
    """
    1. This function should return the solution as instructed in assignment, consisting of a list of moves like ["R2","D2","U1"].
    2. Do not add any extra parameters to the solve() function, or it will break our grading and testing code.
       For testing we will call this function with single argument(initial_board) and it should return 
       the solution.
    3. Please do not use any global variables, as it may cause the testing code to fail.
    4. You can assume that all test cases will be solvable.
    5. The current code just returns a dummy solution.
    """
    #print(initial_board)
    # Reshape the flat representation into a 5x5 matrix
    matrix = [[initial_board[i * COLS + j] for j in range(COLS)] for i in range(ROWS)]

    fringe = []
    closed_states = set()

    # Use a priority queue (heap) instead of a simple list for fringe
    # Each element in the fringe is a tuple (priority, state, path)
    heapq.heappush(fringe, (h(matrix), matrix, []))
    # ...
    while fringe:
        _, state, path = heapq.heappop(fringe)

        state_matrix = tuple(tuple(row) for row in state)  # Convert the state to a hashable tuple of tuples

        if state_matrix in closed_states:
            continue  # Skip this state, it has already been explored

        closed_states.add(state_matrix)  # Add the state matrix to the closed set

        if is_goal(state):
            return path

        for s, move in successors(state):
            heapq.heappush(fringe, (h(s), s, path + [move]))

    #return ["Oc","L2","Icc", "R4"]

# Please don't modify anything below this line
#
if __name__ == "__main__":
    if(len(sys.argv) != 2):
        raise(Exception("Error: expected a board filename"))

    start_state = []
    with open(sys.argv[1], 'r') as file:
        for line in file:
            start_state += [ int(i) for i in line.split() ]

    if len(start_state) != ROWS*COLS:
        raise(Exception("Error: couldn't parse start state file"))

    print("Start state: \n" +"\n".join(printable_board(tuple(start_state))))

    print("Solving...")
    route = solve(tuple(start_state))
    
    print("Solution found in " + str(len(route)) + " moves:" + "\n" + " ".join(route))
           