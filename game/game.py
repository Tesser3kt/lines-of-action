import pygame_sdl2

pygame_sdl2.import_as_pygame()

import pygame as pg
from loguru import logger
from copy import deepcopy
from game.board import Board
from game.objects import GameObject, Stone


class Game:
    """Class for handling high-level game mechanics and graphics."""

    def __init__(self, screen: pg.Surface, config: dict):
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()
        self.cell_width = self.width // 8
        self.cell_height = self.height // 8
        self.config = config
        self.black_stones = {(row, col): None for row in range(8) for col in range(8)}
        self.white_stones = {(row, col): None for row in range(8) for col in range(8)}
        self.tile = None
        self.highlight_rect = None
        self.current_highlight_rects = pg.sprite.RenderUpdates()
        self.cells_to_redraw = set()

    def _init_board(self) -> None:
        """Initialises the game board."""
        logger.debug("Initialising game board...")
        self.board = Board()
        self.board.initialise()

    @logger.catch(message="Drawing board failed.")
    def _draw_board(self) -> None:
        """Draws the game board: fills with green and adds black grid."""
        logger.debug("Drawing board...")
        grid_color = self.config["game"]["board"]["bg_color"]
        line_color = self.config["game"]["board"]["line_color"]

        self.screen.fill(grid_color)

        # Draw vertical lines
        for x in range(0, self.width, self.width // 8):
            pg.draw.line(self.screen, line_color, (x, 0), (x, self.height), 2)

        # Draw horizontal lines
        for y in range(0, self.height, self.height // 8):
            pg.draw.line(self.screen, line_color, (0, y), (self.width, y), 2)

    @logger.catch(message="Failed to draw stone.")
    def _draw_stone(self, row: int, col: int, player: int) -> None: ...

    @logger.catch(message="Error drawing initial stones.")
    def _place_initial_stones(self) -> None:
        logger.debug("Placing initial stones...")

        black_color = self.config["objects"]["stone"]["black"]["color"]
        white_color = self.config["objects"]["stone"]["white"]["color"]
        margin = self.config["objects"]["stone"]["margin"]

        # Place stones to correct coordinates
        for row in range(8):
            for col in range(8):
                if self.board.board[row, col] == 1:
                    new_stone = Stone(
                        color=black_color,
                        width=self.cell_width,
                        height=self.cell_height,
                        margin=margin,
                    )
                    new_stone.move_to(
                        col * self.cell_width,
                        row * self.cell_height,
                    )
                    self.black_stones[(row, col)] = new_stone
                if self.board.board[row, col] == -1:
                    new_stone = Stone(
                        color=white_color,
                        width=self.cell_width,
                        height=self.cell_height,
                        margin=margin,
                    )
                    new_stone.move_to(
                        col * self.cell_width,
                        row * self.cell_height,
                    )
                    self.white_stones[(row, col)] = new_stone

        # Blit stones to screen
        for stone in self.black_stones.values():
            if stone:
                self.screen.blit(stone.image, stone.rect)
        for stone in self.white_stones.values():
            if stone:
                self.screen.blit(stone.image, stone.rect)

    def _highlight_cell(self, row: int, col: int) -> None:
        """Highlights a cell given its board coordinates."""
        highlight_color = self.config["game"]["board"]["highlight_color"]

        highlight_rect = GameObject(
            color=(*highlight_color, 128),
            width=self.cell_width - 4,
            height=self.cell_height - 4,
        )
        highlight_rect.move(col * self.cell_width + 2, row * self.cell_height + 2)
        self.current_highlight_rects.add(highlight_rect)

    def clear_available_moves(self) -> None:
        """Clears highlighted cells with availables moves. Redraws affected stones."""
        bg_color = self.config["game"]["board"]["bg_color"]
        clear_bg = pg.Surface((self.width, self.height))
        clear_bg.fill(tuple(bg_color))

        self.current_highlight_rects.clear(self.screen, clear_bg)

    def _redraw_stones(self, cells: list) -> list[pg.Rect]:
        """Redraws stones given list of cells. Returns affected rects."""
        logger.debug("Redrawing stones from list {}.", cells)

        # Get rects with stones that have to be redrawn
        prev_stone_rects = []
        for rrow, rcol in cells:
            if self.white_stones[(rrow, rcol)]:
                prev_stone_rects.append(self.white_stones[(rrow, rcol)]).rect
            if self.black_stones[(rrow, rcol)]:
                prev_stone_rects.append(self.black_stones[(rrow, rcol)]).rect

    def prepare(self) -> None:
        """Prepares the board for first move."""
        logger.info("Preparing board...")
        self._init_board()
        self._draw_board()
        self._place_initial_stones()

    @logger.catch(message="Failed to move stone.")
    def move_stone(self, source: tuple, target: tuple) -> list[pg.Rect]:
        """Moves a stone and returns a list of updated rects."""
        player = self.board.board[*source]

        margin = self.config["objects"]["stone"]["margin"]
        stone_color = (
            self.config["objects"]["stone"]["black"]["color"]
            if player == 1
            else self.config["objects"]["stone"]["white"]["color"]
        )
        clear_bg = pg.Surface((self.cell_width - 4, self.cell_height - 4))
        clear_bg.fill(self.config["game"]["board"]["bg_color"])

        affected_rects = []
        if not self.board.move_stone(source, target):
            return []

        if player == -1:
            stones = self.white_stones
        else:
            stones = self.black_stones

        # Clear previous cell
        self.screen.blit(
            clear_bg,
            stones[*source].rect.move(2, 2),
        )
        affected_rects.append(stones[*source].rect)

        # Move stone in dict
        stones[*target] = Stone(stone_color, self.cell_width, self.cell_height, margin)
        stones[*target].move_to(
            target[1] * self.cell_width, target[0] * self.cell_height
        )
        stones[*source] = None

        # Blit the stone to new cell
        self.screen.blit(stones[*target].image, stones[*target].rect)
        affected_rects.append(stones[*target].rect)

        return affected_rects

    def highlight_available_moves(self, row: int, col: int) -> list[pg.Rect]:
        logger.debug("Highlighting available moves for stone on ({}, {})", col, row)

        # Clear previously highlighted moves
        self.clear_available_moves()
        self.current_highlight_rects.empty()

        # Calculate available moves
        available_moves = self.board.get_available_moves(row, col)
        self.cells_to_redraw = available_moves

        for target_row, target_col in available_moves:
            self._highlight_cell(target_row, target_col)

        return self.current_highlight_rects.draw(self.screen)
