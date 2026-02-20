import pygame_sdl2

pygame_sdl2.import_as_pygame()

import pygame as pg
from loguru import logger
from copy import deepcopy
from game.board import Board
from game.objects import GameObject, Stone
from ai.ai import AI


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
        self.available_moves = set()

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

    @property
    def _all_stones(self) -> list[tuple[int, Stone]]:
        """Returns all placed stones as tuples (position, stone)."""
        stones = []
        for pos, stone in self.black_stones.items():
            if stone:
                stones.append((pos, stone))
        for pos, stone in self.white_stones.items():
            if stone:
                stones.append((pos, stone))

        return stones

    def _highlight_cell(self, row: int, col: int) -> None:
        """Highlights a cell given its board coordinates."""
        highlight_color = self.config["game"]["board"]["highlight_color"]

        highlight_rect = GameObject(
            color=(*highlight_color, 128),
            width=self.cell_width - 4,
            height=self.cell_height - 4,
        )
        highlight_rect.move_to(col * self.cell_width + 2, row * self.cell_height + 2)
        self.current_highlight_rects.add(highlight_rect)

    def clear_available_moves(self) -> list[pg.Rect]:
        """Clears highlighted cells with availables moves if any. Redraws
        affected stones. Returns list of affected rects."""
        if not self.current_highlight_rects:
            return []

        bg_color = self.config["game"]["board"]["bg_color"]
        clear_bg = pg.Surface((self.width, self.height))
        clear_bg.fill(tuple(bg_color))

        logger.debug(
            "Clearing current highlight rects from {}.", self.current_highlight_rects
        )

        # Get all highlighted rects positions
        rects = [h.rect for h in self.current_highlight_rects]

        # Clear highlighted rects
        self.current_highlight_rects.clear(self.screen, clear_bg)
        self.current_highlight_rects.empty()
        rects += self._redraw_stones(self.available_moves)
        self.available_moves = set()

        return rects

    def _redraw_stones(self, cells: list) -> list[pg.Rect]:
        """Redraws stones given list of cells. Returns affected rects."""
        logger.debug("Redrawing stones from list {}.", cells)

        # Get rects with stones that have to be redrawn
        prev_stone_rects = []
        for rrow, rcol in cells:
            if self.white_stones[(rrow, rcol)] is not None:
                prev_stone_rects.extend(self.move_stone((rrow, rcol), (rrow, rcol)))
            if self.black_stones[(rrow, rcol)] is not None:
                prev_stone_rects.extend(self.move_stone((rrow, rcol), (rrow, rcol)))

        return prev_stone_rects

    def prepare(self) -> None:
        """Prepares the board for first move."""
        logger.info("Preparing board...")
        self._init_board()
        self._draw_board()
        self._place_initial_stones()

    def _redraw_all_stones(self) -> None:
        """Redraws all stones (for debugging purposes only)."""
        # raise NotImplementedError("This shouldn't be used in production!")
        self._redraw_stones([pos for (pos, stone) in self._all_stones])

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
        affected_rects.append(
            pg.Rect(
                stones[*source].rect.x,
                stones[*source].rect.y,
                self.cell_width - 4,
                self.cell_height - 4,
            )
        )

        # Move stone in dict is source != target.
        if source != target:
            stones[*target] = Stone(
                stone_color, self.cell_width, self.cell_height, margin
            )
            stones[*target].move_to(
                target[1] * self.cell_width, target[0] * self.cell_height
            )
            stones[*source] = None

        # Blit the stone to new cell
        self.screen.blit(stones[*target].image, stones[*target].rect)
        affected_rects.append(stones[*target].rect)

        return affected_rects

    @logger.catch(message="Error retrieving stone.")
    def get_stone(self, pos: tuple, player: int) -> Stone or None:
        if player == 1:
            return self.black_stones[pos]
        else:
            return self.white_stones[pos]

    def highlight_available_moves(self, row: int, col: int) -> list[pg.Rect]:
        logger.debug("Highlighting available moves for stone on ({}, {})", col, row)

        # Calculate available moves and save them for later redrawing
        self.available_moves = self.board.get_available_moves(row, col)

        for target_row, target_col in self.available_moves:
            self._highlight_cell(target_row, target_col)

        return self.current_highlight_rects.draw(self.screen)

    def get_clicked_stone_pos(self, pos: tuple) -> tuple[int, int] or None:
        """Returns the board positions of a clicked stone."""
        logger.debug("Getting clicked stone for mouse position {}.", pos)
        for stone_pos, stone in self._all_stones:
            if stone.rect.collidepoint(pos):
                return stone_pos

        return None

    def get_clicked_highlight(self, mouse_pos: tuple) -> tuple[int, int] or None:
        """Returns the position of the clicked highlighted rectangle, if any.
        Otherwise None."""

        board_pos = None

        # Check if the clicked position is highlighted
        for highlight_rect in self.current_highlight_rects:
            if highlight_rect.rect.collidepoint(mouse_pos):
                # Calculate highlight rect position
                board_pos = (
                    highlight_rect.rect.y // self.cell_height,
                    highlight_rect.rect.x // self.cell_width,
                )
                if board_pos in self.available_moves:
                    break

        return board_pos

    def game_over(self) -> int:
        """Calls the 'game_over' method of Board."""
        return self.board.game_over()

    def _draw_connection(
        self, source: tuple, target: tuple, color: tuple[int, int, int]
    ) -> None:
        """Draws a connecting line between two cells."""
        source_center = (
            source[1] * self.cell_width + self.cell_width // 2,
            source[0] * self.cell_height + self.cell_height // 2,
        )
        target_center = (
            target[1] * self.cell_width + self.cell_width // 2,
            target[0] * self.cell_height + self.cell_height // 2,
        )

        pg.draw.line(self.screen, color, source_center, target_center, width=8)

    def _draw_component(self, component: list, color: tuple[int, int, int]):
        """Draws connections between neighbors in a given component."""
        for s1, s2 in component:
            self._draw_connection(s1, s2, color)

    def mark_components(self, player: int, color: tuple[int, int, int]) -> None:
        """Marks in given color all  components of the given player."""
        for component in self.board.get_player_components(player):
            self._draw_component(component, color)

    def play_ai_game(self, repeats: int) -> None:
        """Plays a game of AI against AI."""
        logger.debug("Playing AI game...")
        self.board.initialise()

        running = True
        player = 1

        ai_players = {
            1: AI(self.board, self.config, 1),
            -1: AI(self.board, self.config, -1),
        }

        game_repeats = 0
        while game_repeats < repeats:
            logger.debug("Repeat number {}.", game_repeats)
            self.board.reset()

            logger.debug("Starting game...")
            game_repeats += 1
            running = True
            while running:
                # Reset reward for move
                reward = 0

                logger.warning("Before move: \n" + str(self.board))
                # Predict a move
                source, target = ai_players[player].predict_move()
                self.board.move_stone(source, target)
                logger.warning("After move: \n" + str(self.board))

                # Calculate reward
                reward += ai_players[player].get_cum_distance_reward()
                reward += ai_players[player].get_avail_moves_reward(target)

                logger.debug(
                    "Reward for move ({}, {}) as player {}: {}",
                    source,
                    target,
                    player,
                    reward,
                )

                if (victor := self.game_over()) != 0:
                    logger.debug("Player {} won. Exiting...", victor)
                    running = False
                    # TODO reward for winning

                # Switch players
                player = -player
