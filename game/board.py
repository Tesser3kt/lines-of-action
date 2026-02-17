import numpy as np
from loguru import logger


class Board:
    """Game board class. Supports movement of stones, checks victory conditions and stores board data."""

    def __init__(self):
        # Empty board
        self.board = np.matrix([[0 for _ in range(8)] for _ in range(8)])
        self.components = {"black": [], "white": []}

    def __repr__(self):
        return str(self.board)

    def initialise(self) -> None:
        """Initialises the board to starting game position."""

        # Place black stones to the top and bottom rows
        self.board[0] = [0] + [1] * 6 + [0]
        self.board[-1] = [0] + [1] * 6 + [0]

        # Place white stones to the first and last columns
        self.board[:, 0] = np.array([0] + [-1] * 6 + [0]).reshape(8, 1)
        self.board[:, -1] = np.array([0] + [-1] * 6 + [0]).reshape(8, 1)

        # Add the newly created stones to components
        self.components = {"black": [set(), set()], "white": [set(), set()]}
        for i in range(1, 7):
            self.components["black"][0].add((0, i))
            self.components["black"][1].add((7, i))
            self.components["white"][0].add((i, 0))
            self.components["white"][1].add((0, i))

    def _row_sum(self, row_index: int) -> int:
        return np.sum([abs(x) for x in self.board[row_index]])

    def _col_sum(self, col_index: int) -> int:
        return np.sum([abs(x) for x in self.board[:, col_index]])

    def _down_diag_sum(self, row_index: int, col_index: int) -> int:
        m = min(row_index, col_index)
        col_index -= m
        row_index -= m

        sum = 0
        while col_index < 8 and row_index < 8:
            sum += abs(self.board[row_index, col_index])
            col_index += 1
            row_index += 1

        return sum

    def _up_diag_sum(self, row_index: int, col_index: int) -> int:
        m = min(col_index, 7 - row_index)
        col_index -= m
        row_index += m

        sum = 0
        while col_index < 8 and row_index >= 0:
            sum += abs(self.board[row_index, col_index])
            col_index += 1
            row_index -= 1

        return sum

    def _is_on_board(self, row: int, col: int) -> bool:
        return 0 <= row < 8 and 0 <= col < 8

    def _jumps_over_enemy(self, source: tuple, target: tuple, player: int) -> bool:
        source = np.array(source)
        target = np.array(target)
        cells_in_between = np.linspace(
            start=source, stop=target, num=max(abs(target - source)) + 1, dtype=int
        )[1:-1]

        for cell in cells_in_between:
            row = cell[0]
            col = cell[1]

            if self.board[row, col] == -player:
                return True

        return False

    def get_available_moves(self, row: int, col: int) -> set:
        """Calculates the available moves for the stone on position (col, row)."""
        if self.board[row, col] == 0:
            return set()

        logger.debug("Calculating possible moves for stone on ({}, {}).", col, row)
        player = self.board[row, col]
        available_moves = set()
        possible_moves = set()

        # Add theoretically possible vertical moves
        row_sum = self._row_sum(row)
        possible_moves.add((row, col - row_sum))
        possible_moves.add((row, col + row_sum))

        # Add theoretically possible horizontal moves
        col_sum = self._col_sum(col)
        possible_moves.add((row - col_sum, col))
        possible_moves.add((row + col_sum, col))

        # Add theoretically possible down diagonal moves
        down_diag_sum = self._down_diag_sum(row, col)
        possible_moves.add((row - down_diag_sum, col - down_diag_sum))
        possible_moves.add((row + down_diag_sum, col + down_diag_sum))

        # Add theoretically possible up diagonal moves
        up_diag_sum = self._up_diag_sum(row, col)
        possible_moves.add((row - up_diag_sum, col + up_diag_sum))
        possible_moves.add((row + up_diag_sum, col - up_diag_sum))

        for target_row, target_col in possible_moves:
            if not self._is_on_board(target_row, target_col):
                continue

            if self.board[target_row, target_col] == player:
                continue

            if self._jumps_over_enemy((row, col), (target_row, target_col), player):
                continue

            logger.debug(
                "Stone on ({}, {}) can move to ({}, {}).",
                col,
                row,
                target_col,
                target_row,
            )
            available_moves.add((target_row, target_col))

        return available_moves

    def move_stone(self, source: tuple, target: tuple) -> None:
        """Moves a stone from source to target coordinates. If there is no
        stone on source coordinates, exits. Doesn't check move validity!
        Returns success or failure."""
        logger.debug("Moving stone from {} to {}.", source, target)
        if not self._is_on_board(*source) or not self._is_on_board(*target):
            return False

        if self.board[*source] == 0:
            return False

        self.board[*target] = self.board[*source]
        self.board[*source] = 0

        # TODO update components

        return True
