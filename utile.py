import copy
from tqdm import tqdm
import csv

def is_legal_move(move,board_stat,NgBlackPsWhith):
    ''' Method: is_legal_move
        Parameters: move (tuple)
        Returns: boolean (True if move is legal, False otherwise)
        Does: Checks whether the player's move is legal.

              About input: move is a tuple of coordinates (row, col).
    '''
    MOVE_DIRS = [(-1, -1), (-1, 0), (-1, +1),
             (0, -1),           (0, +1),
             (+1, -1), (+1, 0), (+1, +1)]
    
    if move != () and is_valid_coord(move[0], move[1]) \
       and board_stat[move[0]][move[1]] == 0:
        for direction in MOVE_DIRS:
            if has_tile_to_flip(move, direction,board_stat,NgBlackPsWhith):
                return True
    return False

def get_legal_moves(board_stat,NgBlackPsWhith):
    ''' Method: get_legal_moves
        Parameters: self
        Returns: a list of legal moves that can be made
        Does: Finds all the legal moves the current player can make.
              Every move is a tuple of coordinates (row, col).
    '''
    moves = []
    for row in range(len(board_stat)):
        for col in range(len(board_stat)):
            move = (row, col)
            if is_legal_move(move,board_stat,NgBlackPsWhith):
                moves.append(move)
    return moves

def is_valid_coord(row, col,board_size=8):
    ''' Method: is_valid_coord
        Parameters: self, row (integer), col (integer)
        Returns: boolean (True if row and col is valid, False otherwise)
        Does: Checks whether the given coordinate (row, col) is valid.
              A valid coordinate must be in the range of the board.
    '''
    if 0 <= row < board_size and 0 <= col < board_size:
        return True
    return False

def has_tile_to_flip(move, direction,board_stat,NgBlackPsWhith):
    ''' Method: has_tile_to_flip
        Parameters: move (tuple), direction (tuple)
        Returns: boolean 
                 (True if there is any tile to flip, False otherwise)
        Does: Checks whether the player has any adversary's tile to flip
              with the move they make.

              About input: move is the (row, col) coordinate of where the 
              player makes a move; direction is the direction in which the 
              adversary's tile is to be flipped (direction is any tuple 
              defined in MOVE_DIRS).
    '''
    i = 1
    if is_valid_coord(move[0], move[1]):
        while True:
            row = move[0] + direction[0] * i
            col = move[1] + direction[1] * i
            if not is_valid_coord(row, col) or \
                board_stat[row][col] == 0:
                return False
            elif board_stat[row][col] == NgBlackPsWhith:
                break
            else:
                i += 1
    return i > 1


