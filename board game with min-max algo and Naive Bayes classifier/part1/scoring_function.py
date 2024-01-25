import raichu
import can_attack
def scorer(board,depth,player,current_player, players,N):
    
    myPlayer = players[player].copy()
    oppPlayer = players['w' if player== 'b' else 'b'].copy()
    final_score = 0
    my_weights = [20,50,500]
    opp_weights = [20,50,700]
    
    # Material Weight
    score = 0
    for i in range(N):
        for j in range(N):
            if board[i][j] in myPlayer:
                if board[i][j] == myPlayer[0]: score+=my_weights[0]
                if board[i][j] == myPlayer[1]: score+=my_weights[1]
                if board[i][j] == myPlayer[2]: score+=my_weights[2]
            else:
                if board[i][j] == oppPlayer[0]: score-=opp_weights[0]
                if board[i][j] == oppPlayer[1]: score-=opp_weights[1]
                if board[i][j] == oppPlayer[2]: score-=opp_weights[2]
    final_score+=3*score
    
    # Safety + check if board above it is open to move
    score = 0
    for i in range(N):
        for j in range(N):
            if board[i][j] == myPlayer[0]:
                if raichu.is_safe(board,i,j,N,player,players): 
                    score+=10
                    space_cnt= 0
                    cnt=0
                    if player == 'w':
                        for r in range(i+1,N):
                            for c in range((j-N//3 if j-N//3>=0 else 0), (j+N//3 if j+N//3<=N-1 else N-1)):
                                cnt+=1
                                if board[r][c] == '.':
                                    space_cnt+=1
                    else:
                        for r in range(i-1,-1,-1):
                            for c in range((j-N//3 if j-N//3>=0 else 0), (j+N//3 if j+N//3<=N-1 else N-1)):
                                cnt+=1
                                if board[r][c] == '.':
                                    space_cnt+=1
                    if space_cnt == cnt: score+=100
                else: score-=10
            if board[i][j] == myPlayer[1]:
                if raichu.is_safe(board,i,j,N,player,players): 
                    score+=25
                    space_cnt= 0
                    cnt=0
                    if player == 'w':
                        for r in range(i+1,N):
                            for c in range((j-N//3 if j-N//3>=0 else 0), (j+N//3 if j+N//3<=N-1 else N-1)):
                                cnt+=1
                                if board[r][c] == '.':
                                    space_cnt+=1
                    else:
                        for r in range(i-1,-1,-1):
                            for c in range((j-N//3 if j-N//3>=0 else 0), (j+N//3 if j+N//3<=N-1 else N-1)):
                                cnt+=1
                                if board[r][c] == '.':
                                    space_cnt+=1
                    if space_cnt == cnt: score+=100
                else: score-=25
            if board[i][j] == myPlayer[2]:
                if (i==0 and j == 0) or (i==0 and j == N-1) or (i==N-1 and j == 0) or (i==N-1 and j == N-1): score+=100
                if raichu.is_safe(board,i,j,N,player,players): score+=500
                else: score-=500
            if board[i][j] == oppPlayer[0]:
                if raichu.is_safe(board,i,j,N,player,players): 
                    score-=10
                    space_cnt= 0
                    cnt=0
                    if player == 'b':
                        for r in range(i+1,N):
                            for c in range((j-N//3 if j-N//3>=0 else 0), (j+N//3 if j+N//3<=N-1 else N-1)):
                                cnt+=1
                                if board[r][c] == '.':
                                    space_cnt+=1
                    else:
                        for r in range(i-1,-1,-1):
                            for c in range((j-N//3 if j-N//3>=0 else 0), (j+N//3 if j+N//3<=N-1 else N-1)):
                                cnt+=1
                                if board[r][c] == '.':
                                    space_cnt+=1
                    if space_cnt == cnt: score-=100
                else: score+=10
            if board[i][j] == oppPlayer[1]:
                if raichu.is_safe(board,i,j,N,player,players): 
                    score-=25
                    space_cnt= 0
                    cnt=0
                    if player == 'b':
                        for r in range(i+1,N):
                            for c in range((j-N//3 if j-N//3>=0 else 0), (j+N//3 if j+N//3<=N-1 else N-1)):
                                cnt+=1
                                if board[r][c] == '.':
                                    space_cnt+=1
                    else:
                        for r in range(i-1,-1,-1):
                            for c in range((j-N//3 if j-N//3>=0 else 0), (j+N//3 if j+N//3<=N-1 else N-1)):
                                cnt+=1
                                if board[r][c] == '.':
                                    space_cnt+=1
                    if space_cnt == cnt: score-=100
                else: score+=25
            if board[i][j] == oppPlayer[2]:
                if (i==0 and j == 0) or (i==0 and j == N-1) or (i==N-1 and j == 0) or (i==N-1 and j == N-1): score-=100
                if raichu.is_safe(board,i,j,N,player,players): score-=500
                else: score+=500
    final_score+=2.25*score
    
    # Mobility
    score = 0
    for i in range(N):
        for j in range(N):
            if board[i][j] in myPlayer: 
                if player == 'w':
                    if i>2 and (j == 0 or j == N-1): score+=5
                else: 
                    if i<N-3 and (j == 0 or j == N-1): score+=5
                    
                if board[i][j] == myPlayer[0]:
                    if player == 'w': score+=(i-2)
                    else: score+=(N-3-i)
                if board[i][j] == myPlayer[1]:
                    if player == 'w': score+=(i-1)*2
                    else: score+=(N-2-i)*2
            elif board[i][j] in oppPlayer:
                if player == 'b':   # opponent is w
                    if i>2 and (j == 0 or j == N-1): score-=5
                    if i == N-2: score-=20
                else:
                    if i<N-3 and (j == 0 or j == N-1): score-=5
                    if i == 1: score-=20
                if board[i][j] == oppPlayer[0]:
                    if player == 'b': score-=(i-2)
                    else: score-=(N-3-i)
                if board[i][j] == oppPlayer[1]:
                    if player == 'b': score-=(i-2)*2
                    else: score-=(N-2-i)*2
    final_score+=1.25*score

    # Points for becoming raichu and attacks:
    score = 0
    for i in range(N):
        for j in range(N):
            if board[i][j] != '.':
                possible_attacks = can_attack.can_I_attack(board,i,j,N,player,players)
                if board[i][j] == myPlayer[0]:
                    if player == 'w':
                        if i == N-2: score+=80
                        if i == N-3:
                            score+=10
                            if possible_attacks['b']>0: score+=80
                    else:
                        if i == 1: score+=80
                        if i == 2:
                            score+=10
                            if possible_attacks['w']>0: score+=80
                if board[i][j] == myPlayer[1]:
                    if player == 'w':
                        if i == N-2: score+=80
                        if i == N-3: 
                            score+=100
                            if board[i+1][j] in 'bB' and board[i+2][j] == '.': score+=20
                        if i == N-4:
                            score+=50
                            if board[i+2][j] in 'bB' and board[i+3][j] == '.' and board[i+1][j] == '.': score+=50
                    else:
                        if i == 1: score+=80
                        if i == 2:
                            score+=100
                            if board[i-1][j] in 'bB' and board[i-2][j] == '.': score+=20
                        if i == 3:
                            score+=50
                            if board[i-2][j] in 'bB' and board[i-3][j] == '.' and board[i-1][j] == '.': score+=50
                if board[i][j] == myPlayer[2]:
                    if raichu.is_safe(board,i,j,N,player,players):
                        if possible_attacks[oppPlayer[0]]>0 or possible_attacks[oppPlayer[1]]>0 or possible_attacks[oppPlayer[2]]>0:
                            for idx,attacks in enumerate(possible_attacks):
                                score+= possible_attacks[attacks]*opp_weights[idx]*0.5
                if board[i][j] == oppPlayer[0]:
                    if player == 'b':
                        if i == N-2: score-=80
                        if i == N-3:
                            score-=10
                            if possible_attacks['b']>0: score-=80
                    else:
                        if i == 1: score-=80
                        if i == 2:
                            score-=10
                            if possible_attacks['w']>0: score-=80
                if board[i][j] == oppPlayer[1]:
                    if player == 'b':
                        if i == N-2: score-=80
                        if i == N-3: 
                            score-=100
                            if board[i+1][j] in 'bB' and board[i+2][j] == '.': score-=20
                        if i == N-4:
                            score-=50
                            if board[i+2][j] in 'bB' and board[i+3][j] == '.' and board[i+1][j] == '.': score-=50
                    else:
                        if i == 1: score-=80
                        if i == 2: 
                            score-=100
                            if board[i-1][j] in 'bB' and board[i-2][j] == '.': score-=20
                        if i == 3:
                            score-=50
                            if board[i-2][j] in 'bB' and board[i-3][j] == '.' and board[i-1][j] == '.': score-=50
                if board[i][j] == oppPlayer[2]:
                    if raichu.is_safe(board,i,j,N,player,players):
                        if possible_attacks[myPlayer[0]]>0 or possible_attacks[myPlayer[1]]>0 or possible_attacks[myPlayer[2]]>0:
                            for idx,attacks in enumerate(possible_attacks):
                                score-= possible_attacks[attacks]*opp_weights[idx]*0.5
    final_score+=2.25*score
    
    return final_score-depth*0.1