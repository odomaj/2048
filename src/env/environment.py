from gym import Env, spaces, ActType, ObsType
from game_logic import Board
from typing import Union, Tuple


class Environment(Env):
    action_space: Union[None] = None
    observation_space: Union[None] = None
    reward_range: Union[None] = None
    spec: Union[None] = None
    metadata: Union[None] = None
    np_random: Union[None] = None

    def __init__(self):
        pass

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        pass

    def reset(self):
        pass

    def render(self):
        pass
