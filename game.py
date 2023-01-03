import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
import copy
import time
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn

from utile import get_legal_moves,is_legal_move,has_tile_to_flip

BOARD_SIZE=8

def initialze_board():
    board_stat_init=np.zeros((BOARD_SIZE,BOARD_SIZE))
    board_stat_init[3,3]=+1
    board_stat_init[4,4]=+1
    board_stat_init[3,4]=-1
    board_stat_init[4,3]=-1
    
    return board_stat_init


def input_seq_generator(board_stats_seq,length_seq):
    
    board_stat_init=initialze_board()

    if len(board_stats_seq) >= length_seq:
        input_seq=board_stats_seq[-length_seq:]
    else:
        input_seq=[board_stat_init]
        #Padding starting board state before first index of sequence
        for i in range(length_seq-len(board_stats_seq)-1):
            input_seq.append(board_stat_init)
        #adding the inital of game as the end of sequence sample
        for i in range(len(board_stats_seq)):
            input_seq.append(board_stats_seq[i])
            
    return input_seq

def find_best_move(move1_prob,legal_moves):

    best_move=legal_moves[0]
    max_score=move1_prob[legal_moves[0][0],legal_moves[0][1]]
    for i in range(len(legal_moves)):
        if move1_prob[legal_moves[i][0],legal_moves[i][1]]>max_score:
            max_score=move1_prob[legal_moves[i][0],legal_moves[i][1]]
            best_move=legal_moves[i]
    return best_move

def apply_flip(best_move,board_stat,NgBlackPsWhith):
    
    MOVE_DIRS = [(-1, -1), (-1, 0), (-1, +1),
             (0, -1),           (0, +1),
             (+1, -1), (+1, 0), (+1, +1)]

    for direction in MOVE_DIRS:
        if has_tile_to_flip(best_move, direction,board_stat,NgBlackPsWhith):
            i = 1
            while True:
                row = best_move[0] + direction[0] * i
                col = best_move[1] + direction[1] * i
                if board_stat[row][col] == board_stat[best_move[0], best_move[1]]:
                    break
                else:
                    board_stat[row][col] = board_stat[best_move[0], best_move[1]]
                    i += 1
                    
    return board_stat

    

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

for g in [0,1]:
    # Two rounds would be played
    # First player1 starts game, and then Player2 starts the other game
    if g:
        conf={}
        conf['player1']= sys.argv[1]
        conf['player2']= sys.argv[2]
    else:
        conf={}
        conf['player2']= sys.argv[1]
        conf['player1']= sys.argv[2]    

    board_stat=initialze_board()

    moves_log=""
    board_stats_seq=[]
    pass2player=False

    while not np.all(board_stat) and not pass2player:

        NgBlackPsWhith=-1
        board_stats_seq.append(copy.copy(board_stat))
        if torch.cuda.is_available():
            model = torch.load(conf['player1'])
        else:
            model = torch.load(conf['player1'],map_location=torch.device('cpu'))
        model.eval()

        input_seq_boards=input_seq_generator(board_stats_seq,model.len_inpout_seq)
	#if black is the current player the board should be multiplay by -1
        model_input=np.array([input_seq_boards])*-1
        move1_prob = model(torch.tensor(model_input).float().to(device))
        move1_prob=move1_prob.cpu().detach().numpy().reshape(8,8)

        legal_moves=get_legal_moves(board_stat,NgBlackPsWhith)

        if len(legal_moves)>0:
            
            best_move=find_best_move(move1_prob,legal_moves)
            print(f"Black: {best_move} < from possible move {legal_moves}")

            board_stat[best_move[0],best_move[1]]=NgBlackPsWhith
            moves_log+=str(best_move[0]+1)+str(best_move[1]+1)
            
            board_stat=apply_flip(best_move,board_stat,NgBlackPsWhith)

        else:
            print("Black pass")
            if moves_log[-2:]=="__":
                pass2player=True
            moves_log+="__"


        NgBlackPsWhith=+1
        board_stats_seq.append(copy.copy(board_stat))
        if torch.cuda.is_available():
            model = torch.load(conf['player2'])
        else:
            model = torch.load(conf['player2'],map_location=torch.device('cpu'))
        model.eval()

        input_seq_boards=input_seq_generator(board_stats_seq,model.len_inpout_seq)
        #if black is the current player the board should be multiplay by -1
        model_input=np.array([input_seq_boards])
        move1_prob = model(torch.tensor(model_input).float().to(device))
        move1_prob=move1_prob.cpu().detach().numpy().reshape(8,8)

        legal_moves=get_legal_moves(board_stat,NgBlackPsWhith)


        if len(legal_moves)>0:
            
            best_move = find_best_move(move1_prob,legal_moves)
            print(f"White: {best_move} < from possible move {legal_moves}")
            
            board_stat[best_move[0],best_move[1]]=NgBlackPsWhith
            moves_log+=str(best_move[0]+1)+str(best_move[1]+1)
            
            board_stat=apply_flip(best_move,board_stat,NgBlackPsWhith)

        else:
            print("White pass")
            if moves_log[-2:]=="__":
                pass2player=True
            moves_log+="__"

    board_stats_seq.append(copy.copy(board_stat))
    print("Moves log:",moves_log)

    if np.sum(board_stat)<0:
        print(f"Black {conf['player1']} is winner (with {-1*int(np.sum(board_stat))} points)")
    elif np.sum(board_stat)>0:
        print(f"White {conf['player2']} is winner (with {int(np.sum(board_stat))} points)")
    else:
        print(f"Draw")

    #save game log in gif file
    fig,ax = plt.subplots()
    ims = []
    for i in range(len(board_stats_seq)):
        im = plt.imshow(board_stats_seq[i]*-1,
                        extent=[0.5,8.5,0.5,8.5],
                        cmap='binary',
                        interpolation='nearest')
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)   
    ani.save(f"games/game_2.gif", writer='imagemagick', fps=0.8)
