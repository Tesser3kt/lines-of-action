import numpy as np
from loguru import logger
from random import choice
from game.board import Board
from ai.state import State


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

        logger.debug("UCB for {} is {}.", repr(child), ucb)
        return ucb

    def expand(self) -> "Node":
        logger.debug("Expanding by random action.")

        action = choice(list(self.expandable_moves))
        logger.debug("Action {} taken.", action)

        self.expandable_moves.remove(action)

        child_state = State(player=-self.player, board=self.board)
        child_state.board.move_stone(*action)

        child = Node(
            state=child_state, args=self.args, parent=self, action_taken=action
        )
        self.children.append(child)

        return child

    def simulate(self) -> int:
        victor = self.board.game_over()
        if victor != 0:
            value = -self.state.player
            return value

        rollout_state = self.state.copy()
        rollout_player = self.state.player

        while True:
            valid_moves = self.state.board.get_all_player_moves(rollout_player)
            action = choice(list(valid_moves))
            rollout_state = State(rollout_player, board=rollout_state.board)
            rollout_state.board.move_stone(*action)

            victor = rollout_state.board.game_over()

            if victor == rollout_player:
                value = rollout_player
                return value

            if victor == -rollout_player:
                value = -rollout_player
                return value

            rollout_player = -rollout_player

    def back_propagate(self, value: int):
        self.value_sum += value
        self.visit_count += 1

        value = -value
        if self.parent is not None:
            self.parent.back_propagate(value)


class MCTS:
    """The Monte Carlo Tree search class."""

    def __init__(self, board: Board, args):
        self.board = board
        self.args = args
        self.row_count, self.col_count = board.board.shape
        self.action_size = (self.row_count * self.col_count) * 2

    def _action_to_vector(self, action: tuple) -> np.array:
        """Returns the action taken as a binary vector."""
        source, target = action

        # Add source
        source_matrix = np.zeros(shape=(self.row_count, self.col_count))
        source_matrix[*source] = 1

        # Add target
        target_matrix = np.zeros(shape=(self, row_count, self.col_count))
        target_matrix[*target] = 1

        source_vector = source_matrix.flatten()
        target_vector = target_matrix.flatten()

        return np.append(source_vector, target_vector)

    def most_probable_action(self, action_vector: np.array) -> tuple:
        """Returns the most probable action as a tuple source -> target."""

    def search(self, state: State) -> np.array:
        """Performs given number of searches in the MCT. One search consists of
        1. selection
        2. expansion
        3. simulation
        4. back-propagation

        Returns action probabilities.
        """

        logger.debug(
            "Performing {} of MCT searches starting at state {}",
            self.args["num_searches"],
            state,
        )
        root = Node(board=self.board, args=self.args, state=state)

        for search in range(self.args["num_searches"]):
            node = root

            # 1. Selection
            while node.is_fully_expanded():
                # Go downwards until reaching a non-expanded node.
                node = node.select()

            victor = node.board.game_over()

            # Game ended in a loss (the other player was the last to play)
            if victor != 0:
                value = -node.state.player

            if victor == 0:
                # 2. Expansion
                node = node.expand()

                # 3. Simulation
                value = node.simulate()

            # 4. Back-propagation
            node.back_propagate(value)

        # Calculate action probabilities
        action_probs = np.zeros(self.action_size)
        for child in self.children:
            action_probs += self._action_to_vector(child.action) * child.visit_count

        action_probs /= np.sum(action_probs)
        return action_probs
