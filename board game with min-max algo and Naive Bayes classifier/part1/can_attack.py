def can_I_attack(board,i,j,N,maximizingPlayer,players) -> dict:
    
    if board[i][j] in 'wW@': player = 'w'
    elif board[i][j] in 'bB$': player = 'b'
    
    myPlayer = players[player].copy()
    oppPlayer = players['w' if player == 'b' else 'b'].copy()
    possible_attacks = {piece:0 for piece in oppPlayer}
    
    if board[i][j] == myPlayer[0]:
        if myPlayer[1] == 'w':
            if i<N-2:
                if 0<=j<=N-3:
                    if board[i+1][j+1] == oppPlayer[0]:
                        if board[i+2][j+2] == '.':
                            possible_attacks[oppPlayer[0]]+=1
                if 2<=j<=N-1:
                    if board[i+1][j-1] == oppPlayer[0]:
                        print(board[i+1][j+1])
                        if board[i+2][j-2] == '.':
                            possible_attacks[oppPlayer[0]]+=1
        else:
            if i>1:
                if 0<=j<=N-3:
                    if board[i-1][j+1] == oppPlayer[0]:
                        if board[i-2][j+2] == '.':
                            possible_attacks[oppPlayer[0]]+=1
                if 2<=j<=N-1:
                    if board[i-1][j-1] == oppPlayer[0]:
                        if board[i-2][j-2] == '.':
                            possible_attacks[oppPlayer[0]]+=1
    if board[i][j] == myPlayer[1]:
        if i<=N-3:
            if board[i+1][j] == oppPlayer[0]:
                if board[i+2][j] == '.':
                    possible_attacks[oppPlayer[0]]+=1
            if board[i+1][j] == oppPlayer[1]:
                if board[i+2][j] == '.':
                    possible_attacks[oppPlayer[1]]+=1
            if i+3<=N-1:
                if board[i+2][j] == oppPlayer[0]:
                    if board[i+3][j] == '.':
                        possible_attacks[oppPlayer[0]]+=1
                if board[i+2][j] == oppPlayer[1]:
                    if board[i+3][j] == '.':
                        possible_attacks[oppPlayer[0]]+=1    
    if board[i][j] == myPlayer[2]:
        
        if i+1<N:
            for r in range(i+1,N):
                if board[r][j] in myPlayer:
                    break
                if board[r][j] in oppPlayer and r+1<N and board[r+1][j] == '.':
                    if board[r][j] == oppPlayer[0]:
                        possible_attacks[oppPlayer[0]]+=1
                    if board[r][j] == oppPlayer[1]:
                        possible_attacks[oppPlayer[1]]+=1
                    if board[r][j] == oppPlayer[2]:
                        possible_attacks[oppPlayer[2]]+=1
        
        if i-1>=0:
            for r in range(i-1,-1,-1):
                if board[r][j] in myPlayer:
                    break
                if board[r][j] in oppPlayer and r-1>=0 and board[r-1][j] == '.':
                    if board[r][j] == oppPlayer[0]:
                        possible_attacks[oppPlayer[0]]+=1
                    if board[r][j] == oppPlayer[1]:
                        possible_attacks[oppPlayer[1]]+=1
                    if board[r][j] == oppPlayer[2]:
                        possible_attacks[oppPlayer[2]]+=1
        if j-1>=0:
            for c in range(j-1,-1,-1):
                if board[i][c] in myPlayer:
                    break
                if board[i][c] in oppPlayer and c-1>=0 and board[i][c-1] == '.':
                    if board[i][c] == oppPlayer[0]:
                        possible_attacks[oppPlayer[0]]+=1
                    if board[i][c] == oppPlayer[1]:
                        possible_attacks[oppPlayer[1]]+=1
                    if board[i][c] == oppPlayer[2]:
                        possible_attacks[oppPlayer[2]]+=1
        if j+1<N:
            for c in range(j+1,N):
                if board[i][c] in myPlayer:
                    break
                if board[i][c] in oppPlayer and c+1<N and board[i][c+1] == '.':
                    if board[i][c] == oppPlayer[0]:
                        possible_attacks[oppPlayer[0]]+=1
                    if board[i][c] == oppPlayer[1]:
                        possible_attacks[oppPlayer[1]]+=1
                    if board[i][c] == oppPlayer[2]:
                        possible_attacks[oppPlayer[2]]+=1
        
        r= i+1
        c = j+1
        while r<N and c<N:
            if board[r][c] in myPlayer:
                break
            if board[r][c] in oppPlayer and r+1<N and c+1<N and board[r+1][c+1] == '.':
                if board[r][c] == oppPlayer[0]:
                    possible_attacks[oppPlayer[0]]+=1
                if board[r][c] == oppPlayer[1]:
                    possible_attacks[oppPlayer[1]]+=1
                if board[r][c] == oppPlayer[2]:
                    possible_attacks[oppPlayer[2]]+=1
            r+=1
            c+=1
            
        r= i+1
        c = j-1
        while r<N and c>=0:
            if board[r][c] in myPlayer:
                break
            if board[r][c] in oppPlayer and r+1<N and c-1>=0 and board[r+1][c-1] == '.':
                if board[r][c] == oppPlayer[0]:
                    possible_attacks[oppPlayer[0]]+=1
                if board[r][c] == oppPlayer[1]:
                    possible_attacks[oppPlayer[1]]+=1
                if board[r][c] == oppPlayer[2]:
                    possible_attacks[oppPlayer[2]]+=1
            r+=1
            c-=1
            
        r= i-1
        c = j-1
        while r>=0 and c>=0:
            if board[r][c] in myPlayer:
                break
            if board[r][c] in oppPlayer and r-1>=0 and c-1>=0 and board[r-1][c-1] == '.':
                if board[r][c] == oppPlayer[0]:
                    possible_attacks[oppPlayer[0]]+=1
                if board[r][c] == oppPlayer[1]:
                    possible_attacks[oppPlayer[1]]+=1
                if board[r][c] == oppPlayer[2]:
                    possible_attacks[oppPlayer[2]]+=1
            r-=1
            c-=1
            
        r= i-1
        c = j+1
        while r>=0 and c<0:
            if board[r][c] in myPlayer:
                break
            if board[r][c] in oppPlayer and r-1>=0 and c+1<N and board[r-1][c+1] == '.':
                if board[r][c] == oppPlayer[0]:
                    possible_attacks[oppPlayer[0]]+=1
                if board[r][c] == oppPlayer[1]:
                    possible_attacks[oppPlayer[1]]+=1
                if board[r][c] == oppPlayer[2]:
                    possible_attacks[oppPlayer[2]]+=1
            r-=1
            c+=1
    return possible_attacks
    