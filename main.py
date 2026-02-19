import pygame_sdl2

pygame_sdl2.import_as_pygame()

import pygame as pg
import numpy as np
import json
import pathlib
import os
import sys
import time
from loguru import logger
from game.game import Game
from ai.ai import AI
from game.board import Board


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

    # Play AI random game to find bugs
    board = Board()
    ai = AI(board)

    # ai.play_game()

    while running:
        # store rects to be redrawn this frame
        rects_to_redraw = []

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
                                game.mark_components(
                                    victor, config["game"]["board"]["connection_color"]
                                )
                                time.sleep(10)
                                running = False
                                break

                            logger.debug(
                                "Switching player from {} to {}.", player, -player
                            )
                            player = -player

                    # Otherwise clear any previous highlights
                    rects_to_redraw += game.clear_available_moves()

                    # Get the clicked stone
                    clicked_stone = game.get_clicked_stone_pos(event.pos)
                    if clicked_stone:
                        logger.debug("Stone {} clicked.", clicked_stone)

                        # Highlight available moves
                        if game.board.board[*clicked_stone] == player:
                            logger.debug("Marking stone {} as active.", clicked_stone)
                            active_stone = clicked_stone
                            rects_to_redraw += game.highlight_available_moves(
                                *clicked_stone
                            )
        if running:
            pg.display.update(rects_to_redraw)

        # fps
        clock.tick(config["game"]["fps"])

    pg.quit()


if __name__ == "__main__":
    # initialise logger
    base_dir = pathlib.Path(__file__).parent.resolve()
    logger.add(
        os.path.join(base_dir, "main.log"),
        colorize=True,
        format="<green>{time}</green> <level>{message}</level>",
    )
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time}</green> <level>{message}</level>",
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
    main(config=config)
