import numpy as np
import networkx as nx
from loguru import logger
from collections import deque
from copy import deepcopy


class Board:
    """Game board class. Supports movement of stones, checks victory conditions
    and stores board data."""

    def __init__(self):
        # Empty board
        self.board = np.matrix([[0 for _ in range(8)] for _ in range(8)])
        self.components = {1: [], -1: []}

    def __repr__(self):
        return str(self.board)

    def initialise(self) -> None:
        """Initialises the board to starting game position."""

        # Place black stones to the top and bottom rows
        self.board[0] = [0] + [1] * 6 + [0]
        self.board[-1] = [0] + [1] * 6 + [0]

        # Place white stones to the first and last columns
        self.board[:, 0] = np.array([0] + [-1] * 6 + [0]).reshape(8, 1)
        self.board[:, -1] = np.array([0] + [-1] * 6 + [0]).reshape(8, 1)

        # Add the newly created stones to sets
        self.components = {1: [set(), set()], -1: [set(), set()]}
        for i in range(1, 7):
            self.components[1][0].add((0, i))
            self.components[1][1].add((7, i))
            self.components[-1][0].add((i, 0))
            self.components[-1][1].add((i, 7))

    def _row_sum(self, row_index: int) -> int:
        return np.sum([abs(x) for x in self.board[row_index]])

    def _col_sum(self, col_index: int) -> int:
        return np.sum([abs(x) for x in self.board[:, col_index]])

    def _down_diag_sum(self, row_index: int, col_index: int) -> int:
        m = min(row_index, col_index)
        col_index -= m
        row_index -= m

        sum = 0
        while col_index < 8 and row_index < 8:
            sum += abs(self.board[row_index, col_index])
            col_index += 1
            row_index += 1

        return sum

    def _up_diag_sum(self, row_index: int, col_index: int) -> int:
        m = min(col_index, 7 - row_index)
        col_index -= m
        row_index += m

        sum = 0
        while col_index < 8 and row_index >= 0:
            sum += abs(self.board[row_index, col_index])
            col_index += 1
            row_index -= 1

        return sum

    def get_stones(self, player: int) -> list[tuple[int, int]]:
        """Gets the positions of all stones belonging to 'player'."""

        stones = []
        for row in range(8):
            for col in range(8):
                if self.board[row, col] == player:
                    stones.append((row, col))

        return stones

    def _is_on_board(self, row: int, col: int) -> bool:
        return 0 <= row < 8 and 0 <= col < 8

    def _get_neighbors(self, row: int, col: int) -> set[tuple[int, int]]:
        """Retrieves the board neighbors of a given cell."""
        neighbors = set()
        directions = [
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        ]

        pos = np.array([row, col])
        for direction in directions:
            new_row, new_col = tuple(pos + np.array(direction))
            if self._is_on_board(new_row, new_col):
                neighbors.add((new_row, new_col))

        return neighbors

    def _jumps_over_enemy(self, source: tuple, target: tuple, player: int) -> bool:
        source = np.array(source)
        target = np.array(target)
        cells_in_between = np.linspace(
            start=source, stop=target, num=max(abs(target - source)) + 1, dtype=int
        )[1:-1]

        for cell in cells_in_between:
            row = cell[0]
            col = cell[1]

            if self.board[row, col] == -player:
                return True

        return False

    def get_all_player_moves(
        self, player: int, ignore_collision: bool = False
    ) -> set[tuple[int, int]]:
        """Calculates all the possible moves that a given player can make as tuples source -> target."""
        logger.debug("Calculating all moves for player {}...", player)

        possible_moves = set()
        for row in range(8):
            for col in range(8):
                if self.board[row, col] != player:
                    continue

                for move in self.get_available_moves(row, col, ignore_collision):
                    possible_moves.add(((row, col), move))

        return possible_moves

    def get_available_moves(
        self, row: int, col: int, ignore_collision: bool = False
    ) -> set[tuple[int, int]]:
        """Calculates the available moves for the stone on position (col, row)."""
        if self.board[row, col] == 0:
            return set()

        logger.debug("Calculating possible moves for stone on ({}, {}).", row, col)
        player = self.board[row, col]
        available_moves = set()
        possible_moves = set()

        # Add theoretically possible vertical moves
        row_sum = self._row_sum(row)
        possible_moves.add((row, col - row_sum))
        possible_moves.add((row, col + row_sum))

        # Add theoretically possible horizontal moves
        col_sum = self._col_sum(col)
        possible_moves.add((row - col_sum, col))
        possible_moves.add((row + col_sum, col))

        # Add theoretically possible down diagonal moves
        down_diag_sum = self._down_diag_sum(row, col)
        possible_moves.add((row - down_diag_sum, col - down_diag_sum))
        possible_moves.add((row + down_diag_sum, col + down_diag_sum))

        # Add theoretically possible up diagonal moves
        up_diag_sum = self._up_diag_sum(row, col)
        possible_moves.add((row - up_diag_sum, col + up_diag_sum))
        possible_moves.add((row + up_diag_sum, col - up_diag_sum))

        for target_row, target_col in possible_moves:
            if not self._is_on_board(target_row, target_col):
                continue

            if self.board[target_row, target_col] == player:
                continue

            if (
                self._jumps_over_enemy((row, col), (target_row, target_col), player)
                and not ignore_collision
            ):
                continue

            logger.debug(
                "Stone on ({}, {}) can move to ({}, {}).",
                row,
                col,
                target_row,
                target_col,
            )
            available_moves.add((target_row, target_col))

        return available_moves

    def _check_components(self, player: int) -> bool:
        """Checks the correctness of components for player. Not to be used in
        production."""
        raise NotImplementedError("This shouldn't be used in production!")

        stones = self.get_stones(player)

        stone_components = {}
        for stone in stones:
            stone_components[stone] = 0

        for stone in stones:
            for component in self.components[player]:
                if stone in component:
                    stone_components[stone] += 1

        for stone in stone_components:
            if stone_components[stone] < 1:
                logger.error("Stone ({}, {}) doesn't appear in any component.", *stone)
                raise Exception("A stone doesn't have a component!")
            if stone_components[stone] > 1:
                logger.error(
                    "Stone ({}, {}) appears in more than one component.", *stone
                )
                raise Exception("A stone has too many components!")

    def _update_component(self, component: set, player: int) -> list[set]:
        """Updates the given component, possibly splitting it into more."""
        logger.debug("Updating component {}.", " -- ".join([str(t) for t in component]))
        new_components = []

        # Do BFS on the component, appending a new component whenever q
        # empties until all elements have been added to some component.
        while component:
            current_component = set()
            q = deque()
            first = list(component)[0]
            q.append(first)
            visited = set([first])

            while q:
                row, col = q.popleft()
                logger.debug("Got ({}, {}) from queue.", row, col)
                current_component.add((row, col))

                neighbors = self._get_neighbors(row, col)
                for nrow, ncol in component:
                    if (nrow, ncol) not in neighbors:
                        continue

                    if (nrow, ncol) in visited:
                        continue

                    q.append((nrow, ncol))
                    visited.add((nrow, ncol))

            new_components.append(current_component.copy())
            component = component.difference(current_component)

        logger.debug("The new components are {}.", new_components)
        return new_components

    def _get_component(self, row: int, col: int, player: int) -> int:
        """Returns the index of the component the given stone belongs to.
        Returns -1 if not found (shouldn't happen)."""

        logger.debug("Searching for the component of stone ({}, {})", row, col)

        for index, component in enumerate(self.components[player]):
            if (row, col) in component:
                return index

        logger.error("Component not found for stone ({}, {})", row, col)
        raise Exception(f"Component not found for stone {(row, col)}")

    def _add_stone(self, row: int, col: int, player: int) -> list[int]:
        """Updates components that change as a result of adding this stone.
        Returns a list of indices of components whereto it belongs. These
        should be united into one."""

        logger.debug("Adding stone ({}, {}).", row, col)
        component_indices = set()

        for neighbor in self._get_neighbors(row, col):
            if self.board[*neighbor] != player:
                continue

            index = self._get_component(*neighbor, player)
            component_indices.add(index)

        return component_indices

    def _remove_stone(self, row: int, col: int, player: int) -> None:
        """Removes stone from its component."""

        logger.debug(
            "Removing stone ({}, {}) from the list of components of player {}.",
            row,
            col,
            player,
        )
        for index, component in enumerate(self.components[player]):
            if (row, col) in component:
                component.remove((row, col))

                # If the component was emptied, remove it.
                if len(component) == 0:
                    self.components[player].pop(index)
                    return

                # Otherwise update it.
                new_components = self._update_component(component, player)
                self.components[player][index] = new_components[0]
                for component in new_components[1:]:
                    self.components[player].append(component)

    def _list_component(self, component: set, player: int) -> set:
        """Give a component as a list of connection between neighboring stones."""

        # Find a stone with minimal number of neighbors
        connections = []

        for stone in component:
            for nstone in self._get_neighbors(*stone):
                if self.board[*nstone] == player:
                    connections.append((stone, nstone))

        return connections

    def get_player_components(self, player: int) -> list[list]:
        """Returns components of the given player."""

        player_components = []
        for component in self.components[player]:
            player_components.append(self._list_component(component, player))

        return player_components

    def _remove_components_at(self, indices: set[int], player: int) -> None:
        """Removes components of a given player on specified indices."""

        new_components = []
        for index, component in enumerate(self.components[player]):
            if index in indices:
                continue

            new_components.append(component)
        self.components[player] = new_components

    def _pretty_print_components(self):
        print("-" * 10 + " Black " + "-" * 10)
        for component in self.components[1]:
            print(" -- ".join([str(t) for t in component]))
        print("-" * 10 + " White " + "-" * 10)
        for component in self.components[-1]:
            print(" -- ".join([str(t) for t in component]))

    def move_stone(self, source: tuple, target: tuple) -> None:
        """Moves a stone from source to target coordinates. If there is no
        stone on source coordinates, exits. Doesn't check move validity!
        Returns success or failure."""
        logger.debug("Moving stone from {} to {}.", source, target)

        # Forbid moves out of board bounds
        if not self._is_on_board(*source) or not self._is_on_board(*target):
            return False

        # Forbid illegal moves
        if self.board[*source] == 0:
            return False

        # Do nothing for "static" moves.
        if source == target:
            return True

        # Get active player.
        player = self.board[*source]

        # Remove enemy stone if any.
        removed_enemy_stone = self.board[*target] != 0
        self.board[*target] = 0

        # Update opponent's components if opponent's stone was removed by this move.
        if removed_enemy_stone:
            logger.debug("Enemy stone on {} removed.", target)
            self._remove_stone(*target, -player)

        # Remove player's stone.
        self.board[*source] = 0

        # Update component that changes by removal of this stone
        self._remove_stone(*source, player)

        # Add player's stone to new position
        self.board[*target] = player

        # Merge components that unite as a result of adding this stone
        indices_to_merge = self._add_stone(*target, player)

        # If there are no indices, the stone is isolated
        if len(indices_to_merge) == 0:
            logger.debug("Adding {} to its new component.", target)
            self.components[player].append(set([target]))
        else:
            # Otherwise join the given components
            logger.debug("Joining components {}.", indices_to_merge)
            new_component = set()
            for index in indices_to_merge:
                new_component = new_component.union(
                    self.components[player][index].copy()
                )
            new_component.add(target)
            logger.debug("The new component is {}.", new_component)
            self._remove_components_at(indices_to_merge, player)

            # Add the new stone to the final union
            self.components[player].append(new_component)

        return True

    def game_over(self) -> int:
        """Returns the indicator of the victor, 0 if the game isn't over yet."""
        if len(self.components[1]) == 1:
            return 1
        if len(self.components[-1]) == 1:
            return -1

        return 0

    def reset(self) -> None:
        """Resets the board to initial state."""
        logger.debug("Resetting board...")
        for row in range(8):
            for col in range(8):
                self.board[row, col] = 0

        self.components = {1: [], -1: []}
        self.initialise()

    def _stone_distance(self, s1: tuple[int, int], s2: tuple[int, int]) -> int:
        """Returns the 'generalised Manhattan' distance between two stones."""
        return max(abs(s1[0] - s2[0]), abs(s1[1] - s2[1]))

    def _get_component_distance(self, c1: set, c2: set) -> int:
        """Returns the distance between components as the minimal of the
        distance between their elements."""

        distance = 100
        for s1 in c1:
            for s2 in c2:
                if (test_distance := self._stone_distance(s1, s2)) < distance:
                    distance = test_distance

        return distance

    def cum_distance(self, player: int) -> int:
        """Calculates the cumulative distance between the player's components.
        Only distances between components that are closest are counted.
        Basically length of a minimal spanning tree."""

        logger.debug("Calculating cumulative distance for player {}.", player)
        # Calculate distances between each pair of components
        distances = {
            (i1, i2): self._get_component_distance(c1, c2) if i1 != i2 else 0
            for (i1, c1) in enumerate(self.components[player])
            for (i2, c2) in enumerate(self.components[player])
        }

        # Create graph of components
        comp_graph = nx.Graph()

        # Add edges of weight=distance
        for (i1, i2), d in distances.items():
            comp_graph.add_edge(i1, i2, weight=d)

        spanning_tree = nx.minimum_spanning_tree(comp_graph)
        return sum(data.get("weight") for *_, data in spanning_tree.edges(data=True))

    def _get_interacting_stones(self, stone: tuple) -> list[tuple]:
        """Retrieves all stones that could potentially interact (be on the same diagonal,
        horizontal or vertical line) as the given stone."""

        directions = [
            (0, 1),
            (1, 0),
            (1, 1),
            (-1, 0),
            (0, -1),
            (-1, -1),
            (-1, 1),
            (1, -1),
        ]

        interacting_stones = []
        for dir in directions:
            new_pos = (stone[0] + dir[0], stone[1] + dir[1])
            while self._is_on_board(*new_pos):
                if self.board[*new_pos] != 0:
                    interacting_stones.append(new_pos)
                new_pos = (new_pos[0] + dir[0], new_pos[1] + dir[1])

        return interacting_stones

    def get_blocked_moves(self, stone: int, player: int) -> list[tuple]:
        """Calculates the moves that were blocked by introduction of this 'stone' by 'player'."""

        logger.debug(
            "Calculating blocked moves by the placement of ({}, {}) by {}.",
            *stone,
            player,
        )
        # Temporarily change stone to opponent's
        self.board[*stone] = -player

        # Calculate current available moves for interacting stones
        istones = self._get_interacting_stones(stone)
        total_available_moves = 0
        for istone in istones:
            available_moves = self.get_available_moves(*istone)
            total_available_moves += len(available_moves)

        # Change stone back to player's
        self.board[*stone] = player
        for istone in istones:
            available_moves = self.get_available_moves(*istone)
            total_available_moves -= len(available_moves)

        return total_available_moves

    def copy(self) -> "Board":
        """Returns a copy of this board."""
        new_board = Board()
        new_board.board = self.board.copy()
        new_board.components = {
            1: deepcopy(self.components[1]),
            -1: deepcopy(self.components[-1]),
        }

        return new_board
