import pygame_sdl2

pygame_sdl2.import_as_pygame()

import pygame as pg
import json
import pathlib
import os
import sys
import time
from loguru import logger
from game.game import Game


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

    time.sleep(2)

    rects = game.move_stone((0, 6), (2, 6))
    pg.display.update(rects)

    logger.info("Starting the game...")
    # main loop
    while running:
        # poll for events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                logger.info("Quit signal received. Exiting...")
                running = False

        pg.display.flip()

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
