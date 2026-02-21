import numpy as np
import torch
from collections import deque
from game.board import Board
from ai.state import State


class Agent:
    def __init__(self):
        self.row_count = 8
        self.col_count = 8

        # Action size is just two pairs of coordinates: source -> target
        self.action_size = (self.row_count + self.col_count) * 2

        # State is just two copies of the board: one for player stones and one
        # for enemy stones + the player number
        self.state_size = (self.row_count * self.col_count) * 2 + 1
        self.board = Board()

    def get_next_state(self, state: State, action: tuple, player: int) -> State:
        """Gets the next board state after performing 'action'."""
        source, target = action
        new_state = state.copy()
        new_state.board.move_stone(*action)
        new_state.player = -new_state.player

        return new_state
