import sys
from .game_logic import Board, Move, MoveResult
import tensorflow as tf
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.typing.types import NestedArraySpec
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step
from typing import Union, Callable
import numpy as np


def less_tiles_reward(board: Board) -> np.float32:
    num_tiles = 16
    for tile in board.board:
        if tile == 0:
            num_tiles -= 1
    return num_tiles / 16


def generic_reward(board: Board) -> np.float32:
    return np.sum(board.board) / 128


def hankel_reward(board: Board) -> np.float32:
    """Compute the reward based on the Hankel matrix"""
    # Define the 4x4 Hankel matrix
    hankel_matrix = np.array(
        [
            1,
            1 / 2,
            1 / 4,
            1 / 8,
            1 / 2,
            1 / 4,
            1 / 8,
            1 / 16,
            1 / 4,
            1 / 8,
            1 / 16,
            1 / 32,
            1 / 8,
            1 / 16,
            1 / 32,
            1 / 64,
        ]
    )
    # Calculate the reward as the dot product of the board and Hankel matrix
    return np.dot(board.board, hankel_matrix) / 64


class GameEnvironment(PyEnvironment):
    board: Union[Board, None] = None
    reward_func: Union[Callable[[Board], np.float32], None] = None

    def __init__(
        self,
        handle_auto_reset: bool = False,
        reward_func: Callable[[Board], np.float32] = generic_reward,
    ) -> None:
        super().__init__(handle_auto_reset)
        self.board = Board()  # Initialize the game board
        self.reward_func = reward_func

        # Define the observation and action specs
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4, 4),  # The 4x4 game grid
            dtype=np.int32,
            minimum=0,  # Tile values are non-negative
            maximum=11,  # Maximum tile value in 2048 game
            name="observation",
        )
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name="action"
        )

    def _reset(self) -> time_step.TimeStep:
        """Reset the environment to its initial state"""
        self.board = Board()
        # Reset the board to the initial state
        return time_step.restart(np.array(self.board.grid(), dtype=np.int32))

    def _step(self, action) -> time_step.TimeStep:
        """Take a step based on the action provided"""
        move_result = self.board.make_move(Move(action + 1))

        if move_result == MoveResult.LOST:
            return time_step.termination(
                np.array(self.board.grid(), dtype=np.int32), reward=0
            )
        elif move_result == MoveResult.WON:
            return time_step.termination(
                np.array(self.board.grid(), dtype=np.int32), reward=1.0
            )
        elif move_result == MoveResult.ILLEGAL_MOVE:
            return time_step.transition(
                np.array(self.board.grid(), dtype=np.int32), reward=-1.0
            )
        else:
            # Compute the reward based on the Hankel matrix
            return time_step.transition(
                np.array(self.board.grid(), dtype=np.int32),
                reward=self.reward_func(self.board),
            )

    def observation_spec(self) -> NestedArraySpec:
        """Returns the spec for the observations (game grid)"""
        return self._observation_spec

    def action_spec(self) -> NestedArraySpec:
        """Returns the spec for the possible actions (0-3: up, right, down, left)"""
        return self._action_spec
