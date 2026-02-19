from game.board import Board
from random import choice
from loguru import logger


class AI:
    """AI class for the game."""

    def __init__(self, board: Board):
        self.board = board

    def predict_move(self, player: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Predicts (so far randomly) the next move of the player."""
        logger.debug("Choosing a move for player {}.", player)
        total_possible_moves = []
        for stone in self.board.get_stones(player):
            for move in self.board.get_available_moves(*stone):
                total_possible_moves.append((stone, move))

        chosen_move = choice(total_possible_moves)
        logger.debug("Move {} chosen.", chosen_move)

        return chosen_move

    def play_game(self):
        logger.debug("Playing AI game...")
        self.board.initialise()

        running = True
        player = 1
        logger.debug("Starting game...")
        while running:
            source, target = self.predict_move(player)
            self.board.move_stone(source, target)
            player = -player
            if self.board.game_over() != 0:
                logger.info("Game over. Quitting.")
                running = False
