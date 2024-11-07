import sys
from game_logic import Board, Move, MoveResult
import tensorflow as tf
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.typing.types import NestedArraySpec
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step
from typing import Union
import numpy as np


class GameEnvironment(PyEnvironment):
    board: Union[Board, None] = None

    def __init__(self, handle_auto_reset=False) -> None:
        super().__init__(handle_auto_reset)
        self.board = Board()

    def __init__(self):
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2,),
            dtype=np.int32,
            minimum=0,
            maximum=10,
            name="observation",
        )
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name="action"
        )
        self._state = 0

    def _reset(self):
        self._state = 0
        return time_step.restart(np.array([self._state, 0], dtype=np.int32))

    def _step(self, action):
        if action == 0:
            self._state -= 1
        else:
            self._state += 1
        return time_step.transition(
            np.array([self._state, 0], dtype=np.int32), reward=0.0
        )

    def observation_spec(self) -> NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> NestedArraySpec:
        return self._action_spec


test = GameEnvironment()
