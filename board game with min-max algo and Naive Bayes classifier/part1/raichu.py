#
# raichu.py : Play the game of Raichu
#
# PLEASE PUT YOUR NAMES AND USER IDS HERE!

import sys
import time
from random import random,randint
import scoring_function
def board_to_string(board, N):
    return "\n".join(board[i:i+N] for i in range(0, len(board), N))

def is_safe(board,i,j,N,player, players):
    myPlayer = players[player]
    piece = board[i][j]
    oppPlayer = players['w' if player == 'b' else 'b']
    if piece == 'w':
        for r in range(i+1,N):
            if board[r][j] in myPlayer:
                break
            if board[r][j] == 'B' and abs(r-i)<3:
                if i-1>=0 and board[i-1][j] == '.':
                    return False
            if board[r][j] == '$' and i-1>=0 and board[i-1][j] == '.':
                return False
  
        if i-1>=0:
            for r in range(i-1,-1,-1):
                if board[r][j] in myPlayer:
                    break
                if board[r][j] == '$' and i+1<N and board[i+1][j] == '.':
                    return False

        if j-1>=0:
            for c in range(j-1,-1,-1):
                if board[i][c] in myPlayer:
                    break
                if board[i][c] == 'B' and abs(c-j)<3:
                    if j+1<N and board[i][j+1] == '.':
                        return False
                if board[i][c] == '$' and j+1<N and board[i][j+1] == '.':
                    return False

        if j+1<N:
            for c in range(j+1,N):
                if board[i][c] in myPlayer:
                    break
                if board[i][c] == 'B' and abs(c-j)<3:
                    if j-1<N and board[i][j-1] == '.':
                        return False
                if board[i][c] == '$' and j-1<N and board[i][j-1] == '.':
                    return False
        
        
        r= i+1
        c = j+1
        while r<N and c<N:
            if board[r][c] in myPlayer:
                break
            if board[r][c] == 'b' and abs(c-j)<2:
                if j-1>=0 and i-1>=0 and board[i-1][j-1] == '.':
                    return False
            if board[r][c] == '$' and j-1>=0 and i-1>=0 and board[i-1][j-1] == '.':
                return False
            r+=1
            c+=1
        
        r= i+1
        c = j-1
        while r<N and c>=0:
            if board[r][c] in myPlayer:
                break
            if board[r][c] == 'b' and abs(c-j)<2:
                if j+1<N and i-1>=0 and board[i-1][j+1] == '.':
                    return False
            if board[r][c] == '$' and j+1<N and i-1>=0 and board[i-1][j+1] == '.':
                return False
            r+=1
            c-=1
        r= i-1
        c = j-1
        while r>=0 and c>=0:
            if board[r][c] in myPlayer:
                break
            if board[r][c] == '$' and j+1<N and i+1<N and board[i+1][j+1] == '.':
                return False
            r-=1
            c-=1
            
        r= i-1
        c = j+1
        while r>=0 and c<0:
            if board[r][c] in myPlayer:
                break
            if board[r][c] == '$' and j-1<N and i+1<N and board[i+1][j-1] == '.':
                return False
            r-=1
            c+=1
        
    if piece == 'W':
        for r in range(i+1,N):
            if board[r][j] in myPlayer or board[r][j] == oppPlayer[0]:
                break
            if board[r][j] == 'B' and abs(r-i)<3:
                if i-1>=0 and board[i-1][j] == '.':
                    return False
            if board[r][j] == '$' and i-1>=0 and board[i-1][j] == '.':
                return False
        
        if i-1>=0:
            for r in range(i-1,-1,-1):
                if board[r][j] in myPlayer or board[r][j] == oppPlayer[0]:
                    break
                if board[r][j] == '$' and i+1<N and board[i+1][j] == '.':
                    return False
        if j-1>=0:
            for c in range(j-1,-1,-1):
                if board[i][c] in myPlayer or board[i][c] == oppPlayer[0]:
                    break
                if board[i][c] == 'B' and abs(c-j)<3:
                    if j+1<N and board[i][j+1] == '.':
                        return False
                if board[i][c] == '$' and j+1<N and board[i][j+1] == '.':
                    return False
        if j+1<N:
            for c in range(j+1,N):
                if board[i][c] in myPlayer or board[i][c] == oppPlayer[0]:
                    break
                if board[i][c] == 'B' and abs(c-j)<3:
                    if j-1<N and board[i][j-1] == '.':
                        return False
                if board[i][c] == '$' and j-1<N and board[i][j-1] == '.':
                    return False
        
        r= i+1
        c = j+1
        while r<N and c<N:
            if board[r][c] in myPlayer or board[r][c] == oppPlayer[0]:
                break
            if board[r][c] == '$' and j-1>=0 and i-1>=0 and board[i-1][j-1] == '.':
                return False
            r+=1
            c+=1
            
        r= i+1
        c = j-1
        while r<N and c>=0:
            if board[r][c] in myPlayer or board[r][c] == oppPlayer[0]:
                break
            if board[r][c] == '$' and j+1<N and i-1>=0 and board[i-1][j+1] == '.':
                return False
            r+=1
            c-=1
            
        r= i-1
        c = j-1
        while r>=0 and c>=0:
            if board[r][c] in myPlayer or board[r][c] == oppPlayer[0]:
                break
            if board[r][c] == '$' and j+1<N and i+1<N and board[i+1][j+1] == '.':
                return False
            r-=1
            c-=1
            
        r= i-1
        c = j+1
        while r>=0 and c<0:
            if board[r][c] in myPlayer or board[r][c] == oppPlayer[0]:
                break
            if board[r][c] == '$' and j-1<N and i+1<N and board[i+1][j-1] == '.':
                return False
            r-=1
            c+=1
            
    if piece == '@':
        
        if i+1<N:
            for r in range(i+1,N):
                if board[r][j] in myPlayer or board[r][j] in oppPlayer[:2]:
                    break
                if board[r][j] == '$' and i-1>=0 and board[i-1][j] == '.':
                    return False
        
        if i-1>=0:
            for r in range(i-1,-1,-1):
                if board[r][j] in myPlayer or board[r][j] in oppPlayer[:2]:
                    break
                if board[r][j] == '$' and i+1<N and board[i+1][j] == '.':
                    return False
        if j-1>=0:
            for c in range(j-1,-1,-1):
                if board[i][c] in myPlayer or board[i][c] in oppPlayer[:2]:
                    break
                if board[i][c] == '$' and j+1<N and board[i][j+1] == '.':
                    return False
        if j+1<N:
            for c in range(j+1,N):
                if board[i][c] in myPlayer or board[i][c] in oppPlayer[:2]:
                    break
                if board[i][c] == '$' and j-1<N and board[i][j-1] == '.':
                    return False
        
        r= i+1
        c = j+1
        while r<N and c<N:
            if board[r][c] in myPlayer or board[r][c] in oppPlayer[:2]:
                break
            if board[r][c] == '$' and j-1>=0 and i-1>=0 and board[i-1][j-1] == '.':
                return False
            r+=1
            c+=1
            
        r= i+1
        c = j-1
        while r<N and c>=0:
            if board[r][c] in myPlayer or board[r][c] in oppPlayer[:2]:
                break
            if board[r][c] == '$' and j+1<N and i-1>=0 and board[i-1][j+1] == '.':
                return False
            r+=1
            c-=1
            
        r= i-1
        c = j-1
        while r>=0 and c>=0:
            if board[r][c] in myPlayer or board[r][c] in oppPlayer[:2]:
                break
            if board[r][c] == '$' and j+1<N and i+1<N and board[i+1][j+1] == '.':
                return False
            r-=1
            c-=1
            
        r= i-1
        c = j+1
        while r>=0 and c<0:
            if board[r][c] in myPlayer or board[r][c] in oppPlayer[:2]:
                break
            if board[r][c] == '$' and j-1<N and i+1<N and board[i+1][j-1] == '.':
                return False
            r-=1
            c+=1
    
    if piece == 'b':
        for r in range(i+1,N):
            if board[r][j] in myPlayer:
                break
            if board[r][j] == '@' and i-1>=0 and board[i-1][j] == '.':
                return False
        
        if i-1>=0:
            for r in range(i-1,-1,-1):
                if board[r][j] in myPlayer:
                    break
                
                if board[r][j] == 'W' and abs(r-i)<3:
                    if i-1>=0 and board[i-1][j] == '.':
                        return False
                if board[r][j] == '@' and i+1<N and board[i+1][j] == '.':
                    return False
        if j-1>=0:
            for c in range(j-1,-1,-1):
                if board[i][c] in myPlayer:
                    break
                if board[i][c] == 'W' and abs(c-j)<3:
                    if j+1<N and board[i][j+1] == '.':
                        return False
                if board[i][c] == '@' and j+1<N and board[i][j+1] == '.':
                    return False
        if j+1<N:
            for c in range(j+1,N):
                if board[i][c] in myPlayer:
                    break
                if board[i][c] == 'W' and abs(c-j)<3:
                    if j-1<N and board[i][j-1] == '.':
                        return False
                if board[i][c] == '@' and j-1<N and board[i][j-1] == '.':
                    return False
        
        r= i+1
        c = j+1
        while r<N and c<N:
            if board[r][c] in myPlayer:
                break
            if board[r][c] == '@' and j-1>=0 and i-1>=0 and board[i-1][j-1] == '.':
                return False
            r+=1
            c+=1
            
        r= i+1
        c = j-1
        while r<N and c>=0:
            if board[r][c] in myPlayer:
                break
            if board[r][c] == '@' and j+1<N and i-1>=0 and board[i-1][j+1] == '.':
                return False
            r+=1
            c-=1
        
        r= i-1
        c = j-1
        while r>=0 and c>=0:
            if board[r][c] in myPlayer:
                break
            if board[r][c] == 'w' and abs(c-j)<2:
                if j+1<N and i+1<N and board[i+1][j+1] == '.':
                    return False
            if board[r][c] == '@' and j+1<N and i+1<N and board[i+1][j+1] == '.':
                return False
            r-=1
            c-=1
            
        r= i-1
        c = j+1
        while r>=0 and c<0:
            if board[r][c] in myPlayer:
                break
            if board[r][c] == 'w' and abs(c-j)<2:
                if j-1>=0 and i+1<N and board[i+1][j-1] == '.':
                    return False
            if board[r][c] == '@' and j-1<N and i+1<N and board[i+1][j-1] == '.':
                return False
            r-=1
            c+=1
            
    if piece == 'B':
        for r in range(i+1,N):
            if board[r][j] in myPlayer or board[r][j] == oppPlayer[0]:
                break
            if board[r][j] == 'W' and abs(r-i)<3:
                if i-1>=0 and board[i-1][j] == '.':
                    return False
            if board[r][j] == '@' and i-1>=0 and board[i-1][j] == '.':
                return False
        
        if i-1>=0:
            for r in range(i-1,-1,-1):
                if board[r][j] in myPlayer or board[r][j] == oppPlayer[0]:
                    break
                if board[r][j] == '@' and i+1<N and board[i+1][j] == '.':
                    return False
        if j-1>=0:
            for c in range(j-1,-1,-1):
                if board[i][c] in myPlayer or board[i][c] == oppPlayer[0]:
                    break
                if board[i][c] == 'W' and abs(c-j)<3:
                    if j+1<N and board[i][j+1] == '.':
                        return False
                if board[i][c] == '@' and j+1<N and board[i][j+1] == '.':
                    return False
        if j+1<N:
            for c in range(j+1,N):
                if board[i][c] in myPlayer or board[i][c] == oppPlayer[0]:
                    break
                if board[i][c] == 'W' and abs(c-j)<3:
                    if j-1<N and board[i][j-1] == '.':
                        return False
                if board[i][c] == '@' and j-1<N and board[i][j-1] == '.':
                    return False
        
        r= i+1
        c = j+1
        while r<N and c<N:
            if board[r][c] in myPlayer or board[r][c] == oppPlayer[0]:
                break
            if board[r][c] == '@' and j-1>=0 and i-1>=0 and board[i-1][j-1] == '.':
                return False
            r+=1
            c+=1
            
        r= i+1
        c = j-1
        while r<N and c>=0:
            if board[r][c] in myPlayer or board[r][c] == oppPlayer[0]:
                break
            if board[r][c] == '@' and j+1<N and i-1>=0 and board[i-1][j+1] == '.':
                return False
            r+=1
            c-=1
            
        r= i-1
        c = j-1
        while r>=0 and c>=0:
            if board[r][c] in myPlayer or board[r][c] == oppPlayer[0]:
                break
            if board[r][c] == '@' and j+1<N and i+1<N and board[i+1][j+1] == '.':
                return False
            r-=1
            c-=1
            
        r= i-1
        c = j+1
        while r>=0 and c<0:
            if board[r][c] in myPlayer or board[r][c] == oppPlayer[0]:
                break
            if board[r][c] == '@' and j-1<N and i+1<N and board[i+1][j-1] == '.':
                return False
            r-=1
            c+=1
            
    if piece == '$':
    
        
        if i+1<N:
            for r in range(i+1,N):
                if board[r][j] in myPlayer or board[r][j] in oppPlayer[:2]:
                    break
                if board[r][j] == '@' and i-1>=0 and board[i-1][j] == '.':
                    return False
        
        if i-1>=0:
            for r in range(i-1,-1,-1):
                if board[r][j] in myPlayer or board[r][j] in oppPlayer[:2]:
                    break
                if board[r][j] == '@' and i+1<N and board[i+1][j] == '.':
                    return False
        if j-1>=0:
            for c in range(j-1,-1,-1):
                if board[i][j] in myPlayer or board[i][j] in oppPlayer[:2]:
                    break
                if board[i][c] == '@' and j+1<N and board[i][j+1] == '.':
                    return False
        if j+1<N:
            for c in range(j+1,N):
                if board[i][j] in myPlayer or board[i][j] in oppPlayer[:2]:
                    break
                if board[i][c] == '@' and j-1<N and board[i][j-1] == '.':
                    return False
        
        r= i+1
        c = j+1
        while r<N and c<N:
            if board[r][c] in myPlayer or board[r][c] in oppPlayer[:2]:
                break
            if board[r][c] == '@' and j-1>=0 and i-1>=0 and board[i-1][j-1] == '.':
                return False
            r+=1
            c+=1
            
        r= i+1
        c = j-1
        while r<N and c>=0:
            if board[r][c] in myPlayer or board[r][c] in oppPlayer[:2]:
                break
            if board[r][c] == '@' and j+1<N and i-1>=0 and board[i-1][j+1] == '.':
                return False
            r+=1
            c-=1
            
        r= i-1
        c = j-1
        while r>=0 and c>=0:
            if board[r][c] in myPlayer or board[r][c] in oppPlayer[:2]:
                break
            if board[r][c] == '@' and j+1<N and i+1<N and board[i+1][j+1] == '.':
                return False
            r-=1
            c-=1
            
        r= i-1
        c = j+1
        while r>=0 and c<0:
            if board[r][c] in myPlayer or board[r][c] in oppPlayer[:2]:
                break
            if board[r][c] == '@' and j-1<N and i+1<N and board[i+1][j-1] == '.':
                return False
            r-=1
            c+=1
    return True

def pichu_moves(board, player, i, j, N):
    successors = []
    new_board = [x.copy() for x in board]
    if player == 'w':   #Check if player is white or black
        if i+1<N and j+1<N: #check if the next diagonal is within bounds
            if new_board[i+1][j+1] == 'b' and i+2<N and j+2<N: #check if the next diagonal item is black and we stay in bounds if we skip it
                if new_board[i+2][j+2] == '.': #Check if we skip it to an empty space
                    if i+2 == N-1:             #Check if the position we skipped it in is the last row, if yes convert to Raichu
                        new_board[i][j] = '.'
                        new_board[i+1][j+1] = '.'
                        new_board[i+2][j+2] = '@'
                    else:
                        new_board[i][j] = '.'
                        new_board[i+1][j+1] = '.'
                        new_board[i+2][j+2] = 'w'
                    successors.append([x.copy() for x in new_board.copy()])
                    new_board = [x.copy() for x in board]

            if new_board[i+1][j+1] == '.': #if next diagonal is not a black pichu, check if it is empty
                if i+1 == N-1:               #check if empty cell is last row, if yes convert to Raichu
                    new_board[i][j] = '.'
                    new_board[i+1][j+1] = '@'
                else:                        #If not last row and empty cell, place the pichu
                    new_board[i][j] = '.'
                    new_board[i+1][j+1] = 'w'
                
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
                
        if i+1<N and j-1>=0: # The same thing above but left diagonal
            if new_board[i+1][j-1] == 'b' and i+2<N and j-2>=0:
                if new_board[i+2][j-2] == '.':
                    if i+2 == N-1:
                        new_board[i][j] = '.'
                        new_board[i+1][j-1] = '.'
                        new_board[i+2][j-2] = '@'
                    elif new_board[i+2][j-2] == '.':
                        new_board[i][j] = '.'
                        new_board[i+1][j-1] = '.'
                        new_board[i+2][j-2] = 'w'
                    successors.append([x.copy() for x in new_board.copy()])
                    new_board = [x.copy() for x in board]
            
            if new_board[i+1][j-1] == '.':
                if i+1 == N-1:
                    new_board[i][j] = '.'
                    new_board[i+1][j-1] = '@'
                else:
                    new_board[i][j] = '.'
                    new_board[i+1][j-1] = 'w'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
    else:                                                           #If the player is black execute this
        if i-1>=0 and j+1<N:                                        #Same code as above but for top right diagonal of black
            if new_board[i-1][j+1] == 'w' and i-2>=0 and j+2<N:
                if new_board[i-2][j+2] == '.':
                    if i-2 == 0:
                        new_board[i][j] = '.'
                        new_board[i-1][j+1] = '.'
                        new_board[i-2][j+2] = '$'
                    elif new_board[i-2][j+2] == '.':
                        new_board[i][j] = '.'
                        new_board[i-1][j+1] = '.'
                        new_board[i-2][j+2] = 'b'
                    successors.append([x.copy() for x in new_board.copy()])
                    new_board = [x.copy() for x in board]
            
            if new_board[i-1][j+1] == '.':
                if i-1 == 0:
                    new_board[i][j] = '.'
                    new_board[i-1][j+1] = '$'
                else:
                    new_board[i][j] = '.'
                    new_board[i-1][j+1] = 'b'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
                
        if i-1>=0 and j-1>=0:                                       #Same code as above but for top left diagonal of black
            if new_board[i-1][j-1] == 'w' and i-2>=0 and j-2>=0:
                if new_board[i-2][j-2] == '.':
                    if i-2 == 0:
                        new_board[i][j] = '.'
                        new_board[i-1][j-1] = '.'
                        new_board[i-2][j-2] = '$'
                    elif new_board[i-2][j-2] == '.':
                        new_board[i][j] = '.'
                        new_board[i-1][j-1] = '.'
                        new_board[i-2][j-2] = 'b'
                    successors.append([x.copy() for x in new_board.copy()])
                    new_board = [x.copy() for x in board]
            
            if new_board[i-1][j-1] == '.':
                if i-1 == 0:
                    new_board[i][j] = '.'
                    new_board[i-1][j-1] = '$'
                else:
                    new_board[i][j] = '.'
                    new_board[i-1][j-1] = 'b'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
    return successors
    
def pikachu_moves(board, player, i, j, N):
    successors = []
    new_board = [x.copy() for x in board.copy()]
    if player == 'w':
        if i+1<N:
            if new_board[i+1][j] == '.':
                if i+1 == N-1:
                    new_board[i][j] = '.'
                    new_board[i+1][j] = '@'
                else:
                    new_board[i][j] = '.'
                    new_board[i+1][j] = 'W'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
                
        if i+2<N:
            if new_board[i+1][j] in 'Bb':
                if new_board[i+2][j] == '.':
                    if i+2 == N-1:
                        new_board[i][j] = '.'
                        new_board[i+1][j] = '.'
                        new_board[i+2][j] = '@'
                    else:
                        new_board[i][j] = '.'
                        new_board[i+1][j] = '.'
                        new_board[i+2][j] = 'W'
                    successors.append([x.copy() for x in new_board.copy()])
                    new_board = [x.copy() for x in board] 
            elif new_board[i+2][j] == '.' and new_board[i+1][j] == '.':
                if i+2 == N-1:
                    new_board[i][j] = '.'
                    new_board[i+2][j] = '@'
                else:
                    new_board[i][j] = '.'
                    new_board[i+2][j] = 'W'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            
        if i+3<N:
            if new_board[i+2][j] in 'Bb':
                if new_board[i+3][j] == '.' and new_board[i+1][j] == '.':
                    if i+3 == N-1:
                        new_board[i][j] = '.'
                        new_board[i+2][j] = '.'
                        new_board[i+3][j] = '@'
                    else:
                        new_board[i][j] = '.'
                        new_board[i+2][j] = '.'
                        new_board[i+3][j] = 'W'
                    successors.append([x.copy() for x in new_board.copy()])
                    new_board = [x.copy() for x in board] 
                
        if j+1<N:
            if new_board[i][j+1] == '.':
                new_board[i][j] = '.'
                new_board[i][j+1] = 'W'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
        
        if j+2<N:
            if new_board[i][j+1] in 'Bb':
                if new_board[i][j+2] == '.':
                    new_board[i][j] = '.'
                    new_board[i][j+1] = '.'
                    new_board[i][j+2] = 'W'
                    successors.append([x.copy() for x in new_board.copy()])
                    new_board = [x.copy() for x in board] 
            elif new_board[i][j+2] == '.' and new_board[i][j+1] == '.':
                new_board[i][j] = '.'
                new_board[i][j+2] = 'W'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
                    
        if j+3<N:
            if new_board[i][j+2] in 'Bb':
                if new_board[i][j+3] == '.' and new_board[i][j+1] == '.':
                    new_board[i][j] = '.'
                    new_board[i][j+2] = '.'
                    new_board[i][j+3] = 'W'
                    successors.append([x.copy() for x in new_board.copy()])
                    new_board = [x.copy() for x in board]
            
        if j-1>=0:
            if new_board[i][j-1] == '.':
                new_board[i][j] = '.'
                new_board[i][j-1] = 'W'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            
        if j-2>=0:
            if new_board[i][j-1] in 'Bb':
                if new_board[i][j-2] == '.':
                    new_board[i][j] = '.'
                    new_board[i][j-1] = '.'
                    new_board[i][j-2] = 'W'
                    successors.append([x.copy() for x in new_board.copy()])
                    new_board = [x.copy() for x in board] 
            elif new_board[i][j-2] == '.'  and new_board[i][j-1] == '.':
                new_board[i][j] = '.'
                new_board[i][j-2] = 'W'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            
        if j-3>=0:
            if new_board[i][j-2] in 'Bb':
                if new_board[i][j-3] == '.' and new_board[i][j-1] == '.':
                    new_board[i][j] = '.'
                    new_board[i][j-2] = '.'
                    new_board[i][j-3] = 'W'
                    successors.append([x.copy() for x in new_board.copy()])
                    new_board = [x.copy() for x in board]
    
    else:
        
        if i-1>=0:
            if new_board[i-1][j] == '.':
                if i-1 == 0:
                    new_board[i][j] = '.'
                    new_board[i-1][j] = '$'
                else:
                    new_board[i][j] = '.'
                    new_board[i-1][j] = 'B'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
        
        if i-2>=0:
            if new_board[i-1][j] in 'wW':
                if new_board[i-2][j] == '.':
                    if i-2 == 0:
                        new_board[i][j] = '.'
                        new_board[i-1][j] = '.'
                        new_board[i-2][j] = '$'
                    else:
                        new_board[i][j] = '.'
                        new_board[i-1][j] = '.'
                        new_board[i-2][j] = 'B'
                    successors.append([x.copy() for x in new_board.copy()])
                    new_board = [x.copy() for x in board] 
            elif new_board[i-2][j] == '.' and new_board[i-1][j] == '.':
                if i-2 == 0:
                    new_board[i][j] = '.'
                    new_board[i-2][j] = '$'
                else:
                    new_board[i][j] = '.'
                    new_board[i-2][j] = 'B'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
                
        if i-3>=0:
            if new_board[i-2][j] in 'Ww':
                if new_board[i-3][j] == '.' and new_board[i-1][j] == '.':
                    if i-3 == 0:
                        new_board[i][j] = '.'
                        new_board[i-2][j] = '.'
                        new_board[i-3][j] = '$'
                    else:
                        new_board[i][j] = '.'
                        new_board[i-2][j] = '.'
                        new_board[i-3][j] = 'B'
                    successors.append([x.copy() for x in new_board.copy()])
                    new_board = [x.copy() for x in board]
            
        if j+1<N:
            if new_board[i][j+1] == '.':
                new_board[i][j] = '.'
                new_board[i][j+1] = 'B'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            
        if j+2<N:
            if new_board[i][j+1] in 'Ww':
                if new_board[i][j+2] == '.':
                    new_board[i][j] = '.'
                    new_board[i][j+1] = '.'
                    new_board[i][j+2] = 'B'
                    successors.append([x.copy() for x in new_board.copy()])
                    new_board = [x.copy() for x in board] 
            elif new_board[i][j+2] == '.'  and new_board[i][j+1] == '.':
                new_board[i][j] = '.'
                new_board[i][j+2] = 'B'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
        
        if j+3<N:
            if new_board[i][j+2] in 'Ww':
                if new_board[i][j+3] == '.' and new_board[i][j+1] == '.':
                    new_board[i][j] = '.'
                    new_board[i][j+2] = '.'
                    new_board[i][j+3] = 'B'
                    successors.append([x.copy() for x in new_board.copy()])
                    new_board = [x.copy() for x in board]
                    
        if j-1>=0:
            if new_board[i][j-1] == '.':
                new_board[i][j] = '.'
                new_board[i][j-1] = 'B'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
                
        if j-2>=0:
            if new_board[i][j-1] in 'Ww':
                if new_board[i][j-2] == '.':
                    new_board[i][j] = '.'
                    new_board[i][j-1] = '.'
                    new_board[i][j-2] = 'B'
                    successors.append([x.copy() for x in new_board.copy()])
                    new_board = [x.copy() for x in board] 
            elif new_board[i][j-2] == '.'  and new_board[i][j-1] == '.':
                new_board[i][j] = '.'
                new_board[i][j-2] = 'B'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
                
        if j-3>=0:
            if new_board[i][j-2] in 'Ww':
                if new_board[i][j-3] == '.' and new_board[i][j-1] == '.':
                    new_board[i][j] = '.'
                    new_board[i][j-2] = '.'
                    new_board[i][j-3] = 'B'
                    successors.append([x.copy() for x in new_board.copy()])
                    new_board = [x.copy() for x in board]
        
    return successors

def raichu_moves(board, player, i, j, N):
    successors = []
    new_board = [x.copy() for x in board]
    
    #White
    if player == 'w':
        #Down
        r = i+1
        new_board = [x.copy() for x in board]
        while r<N:
            if new_board[r][j] == '.':
                new_board[i][j] = '.'
                new_board[r][j] = '@'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            else:
                break
            r+=1
            
        if r<N and new_board[r][j] in 'bB$':
            r+=1
            while r<N:
                if new_board[r][j] == '.':
                    new_board[i][j] = '.'
                    new_board[r-1][j] = '.'
                    new_board[r][j] = '@'
                    successors.append([x.copy() for x in new_board.copy()])
                else: break
                r+=1
                
        #Up
        r = i-1
        new_board = [x.copy() for x in board]
        while r>=0:
            if new_board[r][j] == '.':
                new_board[i][j] = '.'
                new_board[r][j] = '@'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            else:
                break
            r-=1
            
        if r>=0 and new_board[r][j] in 'bB$':
            r-=1
            while r>=0:
                if new_board[r][j] == '.':
                    new_board[i][j] = '.'
                    new_board[r+1][j] = '.'
                    new_board[r][j] = '@'
                    successors.append([x.copy() for x in new_board.copy()])
                else: break
                r-=1
            
        #Left
        c = j-1
        new_board = [x.copy() for x in board]
        while c>=0:
            if new_board[i][c] == '.':
                new_board[i][j] = '.'
                new_board[i][c] = '@'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            else:
                break
            c-=1
            
        if c>=0 and new_board[i][c] in 'bB$':
            c-=1
            while c>=0:
                if new_board[i][c] == '.':
                    new_board[i][j] = '.'
                    new_board[i][c+1] = '.'
                    new_board[i][c] = '@'
                    successors.append([x.copy() for x in new_board.copy()])
                else: break
                c-=1
        
        #Right
        c = j+1
        new_board = [x.copy() for x in board]
        while c<N:
            if new_board[i][c] == '.':
                new_board[i][j] = '.'
                new_board[i][c] = '@'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            else:
                break
            c+=1
            
        if c<N and new_board[i][c] in 'bB$':
            c+=1
            while c<N:
                if new_board[i][c] == '.':
                    new_board[i][j] = '.'
                    new_board[i][c-1] = '.'
                    new_board[i][c] = '@'
                    successors.append([x.copy() for x in new_board.copy()])
                else: break
                c+=1
        #RD
        r=i+1
        c=j+1
        new_board = [x.copy() for x in board]
        while r<N and c<N:
            if new_board[r][c] == '.':
                new_board[i][j] = '.'
                new_board[r][c] = '@'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            else:
                break
            r+=1
            c+=1
            
        if r<N and c<N and new_board[r][c] in 'bB$':
            r+=1
            c+=1
            while r<N and c<N:
                if new_board[r][c] == '.':
                    new_board[i][j] = '.'
                    new_board[r-1][c-1] = '.'
                    new_board[r][c] = '@'
                    successors.append([x.copy() for x in new_board.copy()])
                else: break
                r+=1
                c+=1
        #LD
        r=i+1
        c=j-1
        new_board = [x.copy() for x in board]
        while r<N and c>=0:
            if new_board[r][c] == '.':
                new_board[i][j] = '.'
                new_board[r][c] = '@'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            else:
                break
            r+=1
            c-=1
            
        if r<N and c>=0 and new_board[r][c] in 'bB$':
            r+=1
            c-=1
            while r<N and c>=0:
                if new_board[r][c] == '.':
                    new_board[i][j] = '.'
                    new_board[r-1][c+1] = '.'
                    new_board[r][c] = '@'
                    successors.append([x.copy() for x in new_board.copy()])
                else: break
                r+=1
                c-=1
        #RAD
        new_board = [x.copy() for x in board]
        r=i-1
        c=j+1
        new_board = [x.copy() for x in board]
        while r>=0 and c<N:
            if new_board[r][c] == '.':
                new_board[i][j] = '.'
                new_board[r][c] = '@'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            else:
                break
            r-=1
            c+=1

        if r>=0 and c<N and new_board[r][c] in 'bB$':
            r-=1
            c+=1
            while r>=0 and c<N:
                if new_board[r][c] == '.':
                    new_board[i][j] = '.'
                    new_board[r+1][c-1] = '.'
                    new_board[r][c] = '@'
                    successors.append([x.copy() for x in new_board.copy()])
                else:
                    break
                r-=1
                c+=1
        
        #LAD
        r=i-1
        c=j-1
        new_board = [x.copy() for x in board]
        while r>=0 and c>=0:
            if new_board[r][c] == '.':
                new_board[i][j] = '.'
                new_board[r][c] = '@'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            else:
                break
            r-=1
            c-=1

        if r>=0 and c>=0 and new_board[r][c] in 'bB$':
            r-=1
            c-=1
            while r>=0 and c>=0:
                if new_board[r][c] == '.':
                    new_board[i][j] = '.'
                    new_board[r+1][c+1] = '.'
                    new_board[r][c] = '@'
                    successors.append([x.copy() for x in new_board.copy()])
                else:
                    break
                r-=1
                c-=1
    #Black
    else:
        #Down
        r = i+1
        new_board = [x.copy() for x in board]
        while r<N:
            if new_board[r][j] == '.':
                new_board[i][j] = '.'
                new_board[r][j] = '$'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            else:
                break
            r+=1
            
        if r<N and new_board[r][j] in 'wW@':
            r+=1
            while r<N:
                if new_board[r][j] == '.':
                    new_board[i][j] = '.'
                    new_board[r-1][j] = '.'
                    new_board[r][j] = '$'
                    successors.append([x.copy() for x in new_board.copy()])
                else: break
                r+=1
        
                        
        #Up
        r = i-1
        new_board = [x.copy() for x in board]
        while r>=0:
            if new_board[r][j] == '.':
                new_board[i][j] = '.'
                new_board[r][j] = '$'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            else:
                break
            r-=1
            
        if r>=0 and new_board[r][j] in 'wW@':
            r-=1
            while r>=0:
                if new_board[r][j] == '.':
                    new_board[i][j] = '.'
                    new_board[r+1][j] = '.'
                    new_board[r][j] = '$'
                    successors.append([x.copy() for x in new_board.copy()])
                else: break
                r-=1
            
        #Left
        c = j-1
        new_board = [x.copy() for x in board]
        while c>=0:
            if new_board[i][c] == '.':
                new_board[i][j] = '.'
                new_board[i][c] = '$'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            else:
                break
            c-=1
            
        if c>=0 and new_board[i][c] in 'wW@':
            c-=1
            while c>=0:
                if new_board[i][c] == '.':
                    new_board[i][j] = '.'
                    new_board[i][c+1] = '.'
                    new_board[i][c] = '$'
                    successors.append([x.copy() for x in new_board.copy()])
                else: break
                c-=1
        
        #Right
        c = j+1
        new_board = [x.copy() for x in board]
        while c<N:
            if new_board[i][c] == '.':
                new_board[i][j] = '.'
                new_board[i][c] = '$'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            else:
                break
            c+=1
            
        if c<N and new_board[i][c] in 'wW@':
            c+=1
            while c<N:
                if new_board[i][c] == '.':
                    new_board[i][j] = '.'
                    new_board[i][c-1] = '.'
                    new_board[i][c] = '$'
                    successors.append([x.copy() for x in new_board.copy()])
                else: break
                c+=1
        #RD
        r=i+1
        c=j+1
        new_board = [x.copy() for x in board]
        while r<N and c<N:
            if new_board[r][c] == '.':
                new_board[i][j] = '.'
                new_board[r][c] = '$'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            else:
                break
            r+=1
            c+=1
            
        if r<N and c<N and new_board[r][c] in 'wW@':
            r+=1
            c+=1
            while r<N and c<N:
                if new_board[r][c] == '.':
                    new_board[i][j] = '.'
                    new_board[r-1][c-1] = '.'
                    new_board[r][c] = '$'
                    successors.append([x.copy() for x in new_board.copy()])
                else: break
                r+=1
                c+=1
        #LD
        r=i+1
        c=j-1
        new_board = [x.copy() for x in board]
        while r<N and c>=0:
            if new_board[r][c] == '.':
                new_board[i][j] = '.'
                new_board[r][c] = '$'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            else:
                break
            r+=1
            c-=1
            
        if r<N and c>=0 and  new_board[r][c] in 'wW@':
            r+=1
            c-=1
            while r<N and c>=0:
                if new_board[r][c] == '.':
                    new_board[i][j] = '.'
                    new_board[r-1][c+1] = '.'
                    new_board[r][c] = '$'
                    successors.append([x.copy() for x in new_board.copy()])
                else: break
                r+=1
                c-=1
        #RAD
        new_board = [x.copy() for x in board]
        r=i-1
        c=j+1
        new_board = [x.copy() for x in board]
        while r>=0 and c<N:
            if new_board[r][c] == '.':
                new_board[i][j] = '.'
                new_board[r][c] = '$'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            else:
                break
            r-=1
            c+=1

        if r>=0 and c<N and new_board[r][c] in 'wW@':
            r-=1
            c+=1
            while r>=0 and c<N:
                if new_board[r][c] == '.':
                    new_board[i][j] = '.'
                    new_board[r+1][c-1] = '.'
                    new_board[r][c] = '$'
                    successors.append([x.copy() for x in new_board.copy()])
                else:
                    break
                r-=1
                c+=1
        
        #LAD
        r=i-1
        c=j-1
        new_board = [x.copy() for x in board]
        while r>=0 and c>=0:
            if new_board[r][c] == '.':
                new_board[i][j] = '.'
                new_board[r][c] = '$'
                successors.append([x.copy() for x in new_board.copy()])
                new_board = [x.copy() for x in board]
            else:
                break
            r-=1
            c-=1

        if r>=0 and c>=0 and new_board[r][c] in 'wW@':
            r-=1
            c-=1
            while r>=0 and c>=0:
                if new_board[r][c] == '.':
                    new_board[i][j] = '.'
                    new_board[r+1][c+1] = '.'
                    new_board[r][c] = '$'
                    successors.append([x.copy() for x in new_board.copy()])
                else:
                    break
                r-=1
                c-=1 
                
    return successors

def has_player_won(board,N,player):
    for i in range(N):
        for j in range(N):
            if board[i][j] not in player and board[i][j] != '.':
                return False
    return True

def successors(board,N,player):
    successor_states = []
    for i in range(N):
        for j in range(N):
            if player == 'w':
                if board[i][j] == 'w':
                    successor_states.extend(pichu_moves(board, 'w',i,j,N))
                elif board[i][j] == 'W':
                    successor_states.extend(pikachu_moves(board, 'w',i,j,N))
                elif board[i][j] == '@':
                    successor_states.extend(raichu_moves(board,'w', i,j,N))
            else:
                if board[i][j] == 'b':
                    successor_states.extend(pichu_moves(board,'b',i,j,N))
                elif board[i][j] == 'B':
                    successor_states.extend(pikachu_moves(board,'b',i,j,N))
                elif board[i][j] == '$':
                    successor_states.extend(raichu_moves(board,'b',i,j,N))
    return successor_states

def minimax(board, player, current_player, players, depth, depthLimit, N , alpha, beta):    
    if has_player_won(board,N,players[player]):
        return 100000-depth, alpha,beta
    
    if has_player_won(board,N,players['w' if player == 'b' else 'b']):
        return -100000+depth,alpha,beta
    
    if depth == depthLimit:
        my_sc = scoring_function.scorer(board, depth, player, current_player, players,N)
        opp_sc = scoring_function.scorer(board, depth, 'w' if player == 'b' else 'b', current_player, players,N)
        return my_sc,alpha,beta
    
    new_player = 'w' if current_player == 'b' else 'b'
    
    if player == current_player:
        best_value = -float("inf")
        for new_board in successors(board,N,new_player):
            value,alpha,beta = minimax(new_board,player,new_player,players,depth+1,depthLimit,N,alpha,beta)
            best_value = max(best_value,value)
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break
      
        return best_value,alpha,beta
    else:
        best_value = float("inf")
        for new_board in successors(board, N, new_player):
            value,alpha,beta = minimax(new_board,player,new_player,players,depth+1,depthLimit,N,alpha,beta)
            best_value = min(best_value,value)
            beta = min(beta,best_value)
            if beta <= alpha:
                break
        return best_value,alpha,beta
    
def create_board(N, board):
    initial = [['.' for _ in range(N)] for _ in range(N)]
    k = 0
    for i in range(N):
        for j in range(N):
            initial[i][j] = board[k]
            k+=1
    return initial

def find_best_move(board, N, player, timelimit):
    # This sample code just returns the same board over and over again (which
    # isn't a valid move anyway.) Replace this with your code!
    #
    piece_map = {'w': ['w','W','@'], 'b' : ['b','B','$']}
    start = time.time()
    initial_board = create_board(N, board)
    depth = 0
    temp = []
    successor = successors(initial_board,N,player)
    while time.time() <= start+timelimit*0.85:
        best_moves = []
        best_score= -float("inf")
        best_move = None
        alpha = -float("inf")
        beta = float("inf")
        id = 0
        if temp:
            successor = [x[1] for x in temp]
        for new_board in successor:
            value,a,b = minimax(new_board, player, player, piece_map, 0,depth, N, alpha,beta)
            if value > best_score:
                best_score = value
                best_move = ''.join(sum(new_board,[]))
            if time.time() > start+timelimit*0.85:
                break
            id+=1
        depth+=1
        if id == len(successor):
            past_best_moves = best_move
    yield past_best_moves if past_best_moves else None

if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise Exception("Usage: Raichu.py N player board timelimit")
        
    (_, N, player, board, timelimit) = sys.argv
    N=int(N)
    timelimit=float(timelimit)
    if player not in "wb":
        raise Exception("Invalid player.")

    if len(board) != N*N or 0 in [c in "wb.WB@$" for c in board]:
        raise Exception("Bad board string.")

    print("Searching for best move for " + player + " from board state: \n" + board_to_string(board, N))
    print("Here's what I decided:")
    for new_board in find_best_move(board, N, player, timelimit):
        print(new_board)