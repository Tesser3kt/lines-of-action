import numpy as np
import torch
from loguru import logger
from random import choice
from collections import defaultdict
from game.board import Board
from ai.state import State
from ai.model import ResNet
from ai.spg import SPG


class Node:
    def __init__(
        self,
        state: State,
        args,
        parent: "Node" = None,
        action_taken: tuple = None,
        prior: int = 0,
        visit_count: int = 0,
    ):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.args = args

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0
        # self.evaluation = self.state.board.evaluate(self.state.player)

    def __repr__(self):
        return f"Node {self.value_sum} / {self.visit_count} with state, {self.state}"

    def is_fully_expanded(self) -> bool:
        return len(self.children) > 0

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
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2

        ucb = (
            q_value
            + self.args["c"]
            * (np.sqrt(self.visit_count) / (child.visit_count + 1))
            * child.prior
        )

        logger.debug("UCB for {} is {}.", repr(child), ucb)
        return ucb

    def expand(self, policy: np.array) -> "Node":
        logger.debug("Expanding by action determined by policy {}.", policy)
        for action, prob in enumerate(policy):
            if prob <= 0:
                continue

            action = MCTSParallel.index_to_action(action)
            child_state = State(player=1, board=self.state.board)
            child_state.board.move_stone(*action)
            child_state.board.change_perspective()

            child = Node(
                state=child_state,
                args=self.args,
                parent=self,
                action_taken=action,
                prior=prob,
            )
            self.children.append(child)

        return child

    def back_propagate(self, value: int):
        self.value_sum += value
        self.visit_count += 1

        # Change to opponent's value
        value = -value
        if self.parent is not None:
            self.parent.back_propagate(value)


class MCTSParallel:
    """The Monte Carlo Tree search class."""

    def __init__(self, board: Board, model: ResNet, args):
        self.board = board.copy()
        self.args = args
        self.model = model
        self.row_count, self.col_count = board.board.shape
        self.action_size = (self.row_count * self.col_count) ** 2

    def index_to_action(index: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Converts the given index in an array of size 'action_size' to stone
        move. Basically converts to to base 8."""
        oct_string = "{0:04o}".format(index)[::-1]
        source = int(oct_string[0]), int(oct_string[1])
        target = int(oct_string[2]), int(oct_string[3])

        logger.debug("Converted {} to action ({}, {})", index, source, target)
        return (source, target)

    def probs_to_vector(self, probs: dict) -> np.array:
        """Converts the action/value dictionary to a vector."""
        vector = np.zeros(self.action_size)
        for action, prob in probs.items():
            source, target = action
            index = (
                source[0]
                + source[1] * self.row_count
                + target[0] * self.row_count**2
                + target[1] * self.row_count**3
            )
            vector[index] = prob

        return vector

    @torch.no_grad()
    def search(self, states: list[State], selfplay_games: list[SPG]) -> dict:
        """Performs given number of searches in the MCT. One search consists of
        1. selection
        2. expansion
        3. simulation (removed due to model usage)
        4. back-propagation

        Returns action probabilities.
        """

        policy, _ = self.model(
            torch.tensor(State.encode_states(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args["dirichlet_epsilon"]) + self.args[
            "dirichlet_epsilon"
        ] * np.random.dirichlet(
            [self.args["dirichlet_alpha"]] * self.action_size, size=policy.shape[0]
        )

        for i, spg in enumerate(selfplay_games):
            spg_policy = policy[i]
            valid_moves = states[i].board.get_all_player_moves(states[i].player)
            for j, p in enumerate(spg_policy):
                move = MCTSParallel.index_to_action(j)
                if move not in valid_moves:
                    spg_policy[j] = 0

            spg_policy /= np.sum(spg_policy)

            spg.root = Node(state=states[i], args=self.args, visit_count=1)
            spg.root.expand(spg_policy)

        for search in range(self.args["num_searches"]):
            for i, spg in enumerate(selfplay_games):
                logger.debug(
                    "Performing {} of MCT searches starting at state {}.",
                    self.args["num_searches"],
                    spg.state,
                )

                spg.node = None
                node = spg.root

                # Selection
                while node.is_fully_expanded():
                    logger.debug(
                        "Node {} fully expanded, continuing search...", repr(node)
                    )
                    # Go downwards until reaching a non-expanded node.
                    node = node.select()

                victor = node.state.board.game_over()

                # Game ended in a loss (the other player was the last to play)
                if victor != 0:
                    logger.debug("Game over, returning value -1.")
                    value = -node.state.player
                    node.back_propagate(value)
                else:
                    spg.node = node

            # Expand games which haven't terminated.
            expandable_games = [
                index
                for index in range(len(selfplay_games))
                if selfplay_games[index].node is not None
            ]
            if len(expandable_games) > 0:
                states = [
                    selfplay_games[index].node.state for index in expandable_games
                ]
                encoded_states = State.encode_states(states)
                tensor = torch.tensor(encoded_states, device=self.model.device)
                policy, value = self.model(tensor)
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()

            for i, index in enumerate(expandable_games):
                node = selfplay_games[index].node
                spg_policy, spg_value = policy[i], value[i]

                valid_moves = node.state.board.get_all_player_moves(node.state.player)
                for j, p in enumerate(spg_policy):
                    move = MCTSParallel.index_to_action(j)
                    if move not in valid_moves:
                        spg_policy[j] = 0

                spg_policy /= np.sum(spg_policy)
                logger.debug("Given policy is {}.", spg_policy)

                spg_value = spg_value.item()
                logger.debug("Given value is {}.", spg_value)

                # Expansion
                node = node.expand(spg_policy)
                node.back_propagate(spg_value)
