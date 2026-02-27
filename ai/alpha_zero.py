import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from loguru import logger
from collections import defaultdict
from game.board import Board
from ai.state import State
from ai.model import ResNet
from ai.mcts import MCTSParallel
from ai.spg import SPG

ROW_COUNT = 8
COL_COUNT = 8
ACTION_SIZE = (ROW_COUNT * COL_COUNT) ** 2


class AlphaZero:
    """The main AlphaZero class."""

    def __init__(
        self, model: ResNet, board: Board, optimizer: torch.optim.Optimizer, args
    ):
        self.board = board
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.mcts = MCTSParallel(board=board, model=model, args=args)

    def self_play(self) -> list:
        """Pits AlphaZero against itself in a game."""
        # Prepare self play.
        logger.info("Preparing self play...")
        return_memory = []
        player = 1
        selfplay_games = [
            SPG(board=self.board) for _ in range(self.args["num_parallel"])
        ]

        state = State(player=1, board=self.board)
        state.board.change_perspective()

        logger.info("Running game loop...")
        # Game loop
        while len(selfplay_games) > 0:
            # Get all neutral states
            states = [spg.state for spg in selfplay_games]
            neutral_states = []
            for state in states:
                neutral_state = State(player=1, board=state.board)
                neutral_state.board.change_perspective()
                neutral_states.append(neutral_state)

            self.mcts.search(neutral_states, selfplay_games)

            for i in range(len(selfplay_games))[::-1]:
                logger.info("Running game {} / {}...", i, len(selfplay_games))
                spg = selfplay_games[i]

                action_probs = defaultdict(int)
                for child in spg.root.children:
                    action_probs[child.action_taken] += child.visit_count

                prob_sum = sum(action_probs.values())
                for child in spg.root.children:
                    action_probs[child.action_taken] /= prob_sum

                action_probs = self.mcts.probs_to_vector(action_probs)

                # Remember action probs
                spg.memory.append((spg.root.state.copy(), action_probs, player))

                # Choose random action based on probs
                temperature_action_probs = action_probs ** (
                    1 / self.args["temperature"]
                )
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(ACTION_SIZE, p=temperature_action_probs)
                move = MCTSParallel.index_to_action(action)

                logger.info("Move {} chosen for player {}.", move, player)

                # Make the move
                spg.state.board.move_stone(*move)

                # Check game over
                value = spg.state.board.game_over()
                if value != 0:
                    logger.info(
                        "Game {} / {} has ended. Deleting.", i, len(selfplay_games)
                    )
                    return_memory = []
                    for hist_state, hist_probs, hist_player in spg.memory:
                        hist_outcome = value if player == hist_player else -value
                        return_memory.append(
                            ((hist_state.encode()), hist_probs, hist_outcome)
                        )

                    # Delete game
                    del selfplay_games[i]

            # Flip player
            player = -player

        return return_memory

    def train(self, memory: list) -> None:
        random.shuffle(memory)
        for batch_index in range(0, len(memory), self.args["batch_size"]):
            sample = memory[
                batch_index : min(
                    len(memory) - 1, batch_index + self.args["batch_size"]
                )
            ]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = (
                np.array(state),
                np.array(policy_targets),
                np.array(value_targets).reshape(-1, 1),
            )
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(
                policy_targets, dtype=torch.float32, device=self.model.device
            )
            value_targets = torch.tensor(
                value_targets, dtype=torch.float32, device=self.model.device
            )

            out_policy, out_value = self.model(state)
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self) -> None:
        logger.info("Starting learning process based on parameters.")
        for iteration in range(self.args["num_iterations"]):
            logger.info("Learning iteration number {}.", iteration)
            memory = []

            self.model.eval()
            for selfplay_iter in tqdm(
                range(self.args["num_selfplay_iterations"] // self.args["num_parallel"])
            ):
                logger.info("Self-play iteration number {}.", selfplay_iter)
                memory += self.self_play()

            logger.info("Training model.")
            self.model.train()
            for epoch in range(self.args["num_epochs"]):
                logger.info("Training epoch number {}.", epoch)
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")
