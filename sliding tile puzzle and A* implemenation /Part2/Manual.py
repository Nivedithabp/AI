
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
    return [" ".join(['%3d' % cell for cell in row]) for row in board]

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

#
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Error: expected a board filename")

    start_state = []
    with open(sys.argv[1], 'r') as file:
        for line in file:
            start_state.append([int(i) for i in line.split()])

    if len(start_state) != ROWS:
        raise Exception("Error: couldn't parse start state file")

    current_state = start_state
    first_state = start_state

    print("Start state: ")
    for row in printable_board(current_state):
        print(row)

    while True:
        input_str = input("Enter a move and index (e.g., 'Oc 2', 's' for restart, 'q' for quit): ")
        if input_str == 'q':
            break  # Quit the program
        elif input_str == 's':
            current_state = first_state  # Restart from the first state
            print("Restarted to the first state.")
            for row in printable_board(current_state):
                print(row)
        else:
            try:
                move, index_str = input_str.split()
                index = int(index_str)
                if move in ['Oc', 'Occ', 'Ic', 'Icc', 'L', 'R', 'U', 'D'] and 0 <= index <= 4:
                    current_state = apply_move(current_state, move, index)
                    print("Current state: ")
                    for row in printable_board(current_state):
                        print(row)
                else:
                    print("Invalid move or index. Please enter a valid move and index (0 to 4).")
            except ValueError:
                print("Invalid input format. Please enter a move and index (e.g., 'Oc 2').")


