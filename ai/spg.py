from ai.state import State
from game.board import Board


class SPG:
    """Class for storing self-play game data."""

    def __init__(self, board: Board):
        self.board = board
        self.board.initialise()
        self.state = State(player=1, board=self.board)
        self.memory = []
        self.root = None
        self.node = None
