import sys
from .game_logic import Board, Move, MoveResult
import tensorflow as tf
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.typing.types import NestedArraySpec
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step
from typing import Union
import numpy as np


class GameEnvironment(PyEnvironment):
    def __init__(self, handle_auto_reset=False) -> None:
        super().__init__(handle_auto_reset)
        self.board = Board()  # Initialize the game board

        # Define the observation and action specs
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4, 4),  # The 4x4 game grid
            dtype=np.int32,
            minimum=0,  # Tile values are non-negative
            maximum=2048,  # Maximum tile value in 2048 game
            name="observation",
        )
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name="action"
        )
        
    def _reset(self):
        """ Reset the environment to its initial state """
        self.board.reset()  # Reset the board to the initial state
        return time_step.restart(np.array(self.board.board, dtype=np.int32))

    def _step(self, action):
        """ Take a step based on the action provided """
        move_result = self.board.make_move(Move(action))
        
        if move_result == MoveResult.LOST:
            return time_step.termination(np.array(self.board.board, dtype=np.int32), reward=-1.0)
        elif move_result == MoveResult.WON:
            return time_step.termination(np.array(self.board.board, dtype=np.int32), reward=1.0)
        else:
            # Compute the reward based on the Hankel matrix
            reward = self.hankel_reward()
            return time_step.transition(np.array(self.board.board, dtype=np.int32), reward=reward)
    
    def observation_spec(self) -> NestedArraySpec:
        """ Returns the spec for the observations (game grid) """
        return self._observation_spec

    def action_spec(self) -> NestedArraySpec:
        """ Returns the spec for the possible actions (0-3: up, right, down, left) """
        return self._action_spec
    
    def hankel_reward(self):
        """ Compute the reward based on the Hankel matrix """
        grid = np.reshape(self.board.board, (4, 4))

        # Define the 4x4 Hankel matrix
        hankel_matrix = np.array([
            [1, 1/2, 1/4, 1/8],
            [1/2, 1/4, 1/8, 1/16],
            [1/4, 1/8, 1/16, 1/32],
            [1/8, 1/16, 1/32, 1/64]
        ])

        # Calculate the reward as the dot product of the board and Hankel matrix
        reward = np.sum(hankel_matrix * grid)
        return reward

