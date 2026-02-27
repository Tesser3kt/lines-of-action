import pygame_sdl2

pygame_sdl2.import_as_pygame()

import pygame as pg
import numpy as np
import json
import pathlib
import os
import sys
import time
import torch
import matplotlib.pyplot as plt
from loguru import logger
from collections import defaultdict
from game.game import Game
from game.board import Board
from ai.mcts import MCTSParallel
from ai.state import State
from ai.model import ResNet
from ai.alpha_zero import AlphaZero
from ai.spg import SPG


@logger.catch(message="Pygame initialisation failed.")
def _initialise_screen(width: int, height: int) -> pg.Surface:
    """Initialises pygame screen."""
    pg.init()

    return pg.display.set_mode((width, height))


def main(config: dict) -> None:
    """Main function."""
    width = config["screen"]["width"]
    height = config["screen"]["height"]

    screen = _initialise_screen(width, height)
    logger.debug("Screen initialised.")

    clock = pg.time.Clock()
    running = True

    logger.info("Pygame ready.")

    game = Game(screen=screen, config=config)
    game.prepare()
    pg.display.flip()

    logger.info("Starting the game...")
    # main loop

    # Current player
    player = 1

    # currently clicked stone
    active_stone = None

    model = ResNet(
        board=game.board,
        num_resblocks=9,
        num_hidden=128,
        device=config["agent"]["device"],
    )
    model.load_state_dict(
        torch.load("model_1.pt", map_location=config["agent"]["device"])
    )
    model.eval()

    args = {
        "c": 2,
        "num_searches": 25,
        "temperature": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.1,
    }
    mcts = MCTSParallel(board=game.board, model=model, args=args)

    while running:
        # store rects to be redrawn this frame
        rects_to_redraw = []

        # AI plays
        if player == -1:
            # Get action probs
            neutral_state = State(player=1, board=game.board)
            neutral_state.board.change_perspective()
            spg = SPG(board=game.board.copy())
            mcts.search(states=[neutral_state], selfplay_games=[spg])

            action_probs = defaultdict(int)
            for child in spg.root.children:
                action_probs[child.action_taken] += child.visit_count

            prob_sum = sum(action_probs.values())
            for child in spg.root.children:
                action_probs[child.action_taken] /= prob_sum

            action = max(action_probs.items(), key=lambda x: x[1])[0]

            valid_moves = game.board.get_all_player_moves(player)
            if action not in valid_moves:
                logger.error("Action {} not permitted.", action)
                continue

            rects_to_redraw += game.move_stone(*action)

            # Check possible victory
            if (victor := game.game_over()) != 0:
                logger.debug("Player {} has won the game!", victor)
                game.mark_components(
                    victor,
                    config["game"]["board"]["connection_color"],
                )
                pg.display.flip()
                time.sleep(5)
                running = False

            logger.debug("Switching player from {} to {}.", player, -player)
            player = -player
            print(game.board)

        # Human plays
        if player == 1:
            # poll for events
            for event in pg.event.get():
                # Check quit event
                if event.type == pg.QUIT:
                    logger.info("Quit signal received. Exiting...")
                    running = False
                # Check mouse click
                if event.type == pg.MOUSEBUTTONDOWN:
                    logger.debug("Mouse click signal received.")
                    if event.button == 1:
                        # If a stone is active check if player clicked any of the
                        # highlighted positions.
                        if active_stone:
                            clicked_highlight = game.get_clicked_highlight(event.pos)
                            if clicked_highlight:
                                logger.info(
                                    "Performing stone move from {} to {}",
                                    active_stone,
                                    clicked_highlight,
                                )
                                rects_to_redraw += game.move_stone(
                                    active_stone, clicked_highlight
                                )
                                active_stone = None

                                # Check possible victory
                                if (victor := game.game_over()) != 0:
                                    logger.debug("Player {} has won the game!", victor)
                                    game.clear_available_moves()
                                    game.mark_components(
                                        victor,
                                        config["game"]["board"]["connection_color"],
                                    )
                                    pg.display.flip()
                                    time.sleep(5)
                                    running = False
                                    break

                                logger.debug(
                                    "Switching player from {} to {}.", player, -player
                                )
                                print(game.board)
                                player = -player

                        # Otherwise clear any previous highlights
                        rects_to_redraw += game.clear_available_moves()

                        # Get the clicked stone
                        clicked_stone = game.get_clicked_stone_pos(event.pos)
                        if clicked_stone:
                            logger.debug("Stone {} clicked.", clicked_stone)

                            # Highlight available moves
                            if game.board.board[*clicked_stone] == player:
                                logger.debug(
                                    "Marking stone {} as active.", clicked_stone
                                )
                                active_stone = clicked_stone
                                rects_to_redraw += game.highlight_available_moves(
                                    *clicked_stone
                                )
        if running:
            pg.display.update(rects_to_redraw)

        # fps
        clock.tick(config["game"]["fps"])

    pg.quit()


def main_ai(config: dict) -> None:
    player = 1
    board = Board()

    args = {
        "c": 2,
        "num_searches": 100,
        "num_iterations": 4,
        "num_selfplay_iterations": 2000,
        "num_epochs": 5,
        "num_parallel": 500,
        "batch_size": 128,
        "temperature": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.1,
    }
    model = ResNet(board=board, num_resblocks=9, num_hidden=128, device="cuda")
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    az = AlphaZero(model=model, board=board, optimizer=optimizer, args=args)
    az.learn()


if __name__ == "__main__":
    # initialise logger
    base_dir = pathlib.Path(__file__).parent.resolve()
    logger.remove()
    logger.add(
        os.path.join(base_dir, "main.log"),
        colorize=True,
        format="<green>{time}</green> <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time}</green> <level>{message}</level>",
        level="INFO",
    )

    # set numpy to legacy print
    np.set_printoptions(legacy="1.25")

    # try to parse config
    try:
        config_path = os.path.join(base_dir, "config.json")
        config_file = open(config_path, "r", encoding="utf-8")
        config = json.load(config_file)
        config_file.close()
    except IOError or json.JSONDecodeError as e:
        logger.error("Error reading config file: {}", e)

    logger.debug("Config parsed.")

    # Setup torch
    logger.debug("Setting up torch.")

    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # main(config=config)
    main_ai(config=config)
