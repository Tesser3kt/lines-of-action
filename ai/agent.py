import numpy as np
import torch
from collections import deque
from ai.ai import AI
from board.board import Board


class Agent:
    def __init__(self):
        self.row_count = 8
        self.col_count = 8

        # Action size is just two pairs of coordinates: source -> target
        self.action_size = (self.row_count + self.col_count)  * 2

        # State is just two copies of the board: one for player stones and one for enemy stones
        self.state_size = (self.row_count * self.col_count) * 2

        self.board = Board()

    def _state_to_vector(self, board: np.matrix):
        """ Converts the given 8x8 matrix to boolean vector."""
        black_board = np.zeros((self.row_count, self.col_count), dtype=int)
        white_board = np.zeros((self.row_count, self.col_count), dtype=int)

        for row in range(self.row_count):
            for col in range(self.col_count):
                if board[row, col] == 1:
                    black_board[row, col] = 1
                elif board[row, col] == -1:
                    white_board[row, col] = 1

        total_state = black_board.flatten().append(white_board.flatten())
        return total_state

    def get_next_state(self, state: np.matrix, action: tuple, player: int) -> np.matrix:
        """ Gets the next board state after performing 'action'. """
        source, target = action
        self.board.move_stone(source, target)
        return self.board.board
