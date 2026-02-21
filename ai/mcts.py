import numpy as np
from game.board import Board
from ai.state import State
from loguru import logger


class Node:
    def __init__(self, state: State, args, parent=None, action_taken=None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.args = args

        self.children = []
        self.expandable_moves = self.state.get_all_player_moves(self.state.player)

        self.visit_count = 0
        self.value_sum = 0

    def __repr__(self):
        return f"Node {self.value_sum} / {self.visit_count} with state, {self.state}"

    def is_fully_expanded(self) -> bool:
        return len(self.expandable_moves) == 0 and len(self.children) > 0

    def select(self) -> "Node":
        best_child = None
        best_ucb = -np.inf

        logger.debug("Selecting the best child of node {}.", repr(self))
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        logger.debug("Selected {} child as best.", repr(best_child))
        return best_child

    def get_ucb(self, child: "Node") -> float:
        """Calculate the UCB for a given child node."""
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        ucb = (
            q_value
            + self.args["c"]
            + np.sqrt(np.log(self.visit_count) / child.visit_count)
        )

        logger.debug("UCB for {} is {}", repr(child), ucb)
        return ucb

    def expand(self) -> "Node":
        logger.debug("Expanding by random action.")

        action = np.random.choice(self.expandable_moves)
        logger.debug("Action {} taken.", action)

        self.expandable_moves.remove(action)

        child_state = State(player=-self.player, board=self.board)
        child_state.state.move_stone(*action)

        child = Node(
            state=child_state, args=self.args, parent=self, action_taken=action
        )
        self.children.append(child)

        return child


class MCTS:
    """The Monte Carlo Tree search class."""

    def __init__(self, board: Board, args):
        self.board = board
        self.args = args

    def search(self, state):
        """Performs given number of searches in the MCT. One search consists of
        1. selection
        2. expansion
        3. simulation
        4. back-propagation
        """

        logger.debug(
            "Performing {} of MCT searches starting at state {}",
            self.args["num_searches"],
            state,
        )
        root = Node(board=self.board, args=self.args, state=state)

        for search in range(self.args["num_searches"]):
            node = root

            # 1. selection
            while node.is_fully_expanded():
                # Go downwards until reaching a non-expanded node.
                node = node.select()

            victor = node.board.game_over()

            # Game ended in a loss (the other player was the last to play)
            if victor != 0:
                value = -1

            if victor == 0:
                node = node.expand()
