import numpy as np
from game.board import Board


class State:
    """Class for the state of the board. Basically (player, board)."""

    def __init__(self, player: int, board: Board):
        self.player = player
        self.board = board.copy()
        self.row_count, self.col_count = board.board.shape

    def to_vector(self) -> np.array:
        """Converts the given 8x8 matrix to boolean vector. Prepends the player number."""
        total_state = np.array([0]) if self.player == 1 else np.array([1])
        black_board = np.zeros((self.row_count, self.col_count), dtype=int)
        white_board = np.zeros((self.row_count, self.col_count), dtype=int)

        for row in range(self.row_count):
            for col in range(self.col_count):
                if self.board.board[row, col] == 1:
                    black_board[row, col] = 1
                elif self.board.board[row, col] == -1:
                    white_board[row, col] = 1

        total_state.append(black_board.flatten().append(white_board.flatten()))
        return total_state

    def copy(self) -> "State":
        """Returns a copy of this game state."""
        return State(self.player, self.board)

    def __repr__(self):
        return f"{self.player}\n{self.board.board}"

    def encode(self) -> np.array:
        """Encodes the state for PyTorch."""
        encoded_state = np.array(
            (self.board.board == -1, self.board.board == 0, self.board.board == 1)
        ).astype(np.float32)

        return encoded_state

    def encode_states(states: list["State"]) -> np.array:
        """Encodes multiple states as single numpy array."""
        return np.array([state.encode() for state in states]).astype(np.float32)
