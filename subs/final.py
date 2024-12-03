from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.typing.types import NestedArraySpec
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step
from typing import Union, Callable
from enum import Enum
from random import randint
from tf_keras.optimizers import Adam
from tf_agents.networks.q_network import QNetwork
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import (
    TFUniformReplayBuffer,
)
from tf_agents.trajectories import from_transition
from tf_agents.utils import common
from pathlib import Path
import numpy as np
import tensorflow as tf


class Move(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class State(Enum):
    PLAYING = 1
    WON = 2
    LOST = 3


class MoveResult(Enum):
    WON = 1
    LOST = 2
    ILLEGAL_MOVE = 3
    CONTINUE = 4


class Board:
    board: Union[np.ndarray[np.float64], None] = None
    """holds game board"""
    """
    0  1  2  3
    4  5  6  7
    8  9  10 11
    12 13 14 15
    """

    def __init__(self) -> None:
        self.board = np.zeros(16)
        self.add_new()

    def make_move(self, move: Move) -> MoveResult:
        if not self.move(move):
            return MoveResult.ILLEGAL_MOVE
        self.add_new()
        state = self.state()
        if state == State.PLAYING:
            return MoveResult.CONTINUE
        elif state == State.LOST:
            return MoveResult.LOST
        elif state == MoveResult.WON:
            return MoveResult.WON

    def score(self) -> np.float64:
        score = 0
        for i in range(16):
            score += self.value(i)
        return score

    def largest_tile(self) -> np.float64:
        return self.value(np.argmax(self.board))

    def value(self, index: int) -> np.float64:
        if self.board[index] == 0:
            return 0
        return np.power(2, self.board[index])

    def state(self) -> State:
        for row in range(0, 16, 4):
            for col in range(4):
                if self.board[row + col] == 11:
                    return State.WON
        if (
            self.up_legal()
            or self.down_legal()
            or self.left_legal()
            or self.right_legal()
        ):
            return State.PLAYING
        return State.LOST

    def move(self, move: Move) -> bool:
        """calls the correct move function and outputs True if the move was
        successful"""
        if move == Move.UP:
            return self.move_up()
        elif move == Move.DOWN:
            return self.move_down()
        elif move == Move.LEFT:
            return self.move_left()
        elif move == Move.RIGHT:
            return self.move_right()

    def shift_gen(self, compressing: range, stagnant: range) -> bool:
        """general shift tiles"""
        changed: bool = False
        changed_recently: bool = True
        while changed_recently:
            changed_recently = False
            for stag in stagnant:
                last_empty: int = 0
                available_empty: bool = False
                for com in compressing:
                    if self.board[stag + com] != 0 and available_empty:
                        self.board[last_empty] = self.board[stag + com]
                        self.board[stag + com] = 0
                        available_empty = False
                        changed = True
                        changed_recently = True
                    elif self.board[stag + com] == 0 and not available_empty:
                        available_empty = True
                        last_empty = stag + com

        return changed

    def combine_gen(self, compressing: range, stagnant: range) -> bool:
        """general combine tiles"""
        changed: bool = False
        compressing_list: list = list(compressing)
        compress_gap: int = compressing_list[1] - compressing_list[0]
        for stag in stagnant:
            for com in compressing_list[:-1]:
                if (
                    self.board[stag + com] != 0
                    and self.board[stag + com]
                    == self.board[stag + com + compress_gap]
                ):
                    self.board[stag + com + compress_gap] = 0
                    self.board[stag + com] += 1
                    changed = True
        return changed

    def move_gen(self, compressing: range, stagnant: range) -> bool:
        """general move"""
        changed: bool = self.shift_gen(compressing, stagnant)
        if self.combine_gen(compressing, stagnant):
            changed = True
            self.shift_gen(compressing, stagnant)
        return changed

    def move_up(self) -> bool:
        return self.move_gen(range(0, 16, 4), range(4))

    def move_down(self) -> bool:
        return self.move_gen(range(12, -1, -4), range(4))

    def move_left(self) -> bool:
        return self.move_gen(range(4), range(0, 16, 4))

    def move_right(self) -> bool:
        return self.move_gen(range(3, -1, -1), range(0, 16, 4))

    def gen_legal(self, compressing: range, stagnant: range) -> bool:
        """general legal check"""
        for stag in stagnant:
            # look for a zero tile followed by a nonzero tile
            empty_tile: bool = False
            for com in compressing:
                if self.board[stag + com] == 0 and not empty_tile:
                    empty_tile = True
                elif self.board[stag + com] != 0 and empty_tile:
                    return True

            # look for two tiles that can be combined
            compressing_list: list = list(compressing)
            compress_gap: int = compressing_list[1] - compressing_list[0]
            for com in compressing_list[:-1]:
                if (
                    self.board[stag + com]
                    == self.board[stag + com + compress_gap]
                ):
                    return True
        return False

    def up_legal(self) -> bool:
        return self.gen_legal(range(0, 16, 4), range(4))

    def down_legal(self) -> bool:
        return self.gen_legal(range(12, -1, -4), range(4))

    def left_legal(self) -> bool:
        return self.gen_legal(range(4), range(0, 16, 4))

    def right_legal(self) -> bool:
        return self.gen_legal(range(3, -1, -1), range(0, 16, 4))

    def add_new(self) -> bool:
        """adds a random 2 tile on the board if the board is not full
        True if there are no 0 tiles on the board before the add, otherwise
        False"""
        if self.full_board():
            return False
        i = randint(0, 15)
        while self.board[i] != 0:
            i = randint(0, 15)
        if randint(0, 9) == 9:
            self.board[i] = 2
        else:
            self.board[i] = 1
        return True

    def full_board(self) -> bool:
        """True if there are no 0 tiles on the board, otherwise False"""
        for tile in self.board:
            if tile == 0:
                return False
        return True

    def grid(self) -> np.ndarray[np.ndarray[np.float64]]:
        return np.array(
            [
                self.board[:4],
                self.board[4:8],
                self.board[8:12],
                self.board[12:],
            ],
            dtype=np.float64,
        )

    def pretty_tile(self, index: int) -> str:
        val = str(int(self.value(index)))
        if len(val) == 1:
            return f"  {val}   "
        elif len(val) == 2:
            return f"  {val}  "
        elif len(val) == 3:
            return f" {val}  "
        elif len(val) == 4:
            return f" {val} "

    def pretty(self) -> str:
        top: str = "_" * 29
        bottom: str = (("|" + ("_" * 6)) * 4) + "|"
        middle: str = (("|" + (" " * 6)) * 4) + "|"
        for i in range(0, 16, 4):
            row = f"|{self.pretty_tile(i)}|{self.pretty_tile(i+1)}|{self.pretty_tile(i+2)}|{self.pretty_tile(i+3)}|"
            top += f"\n{middle}\n{row}\n{bottom}"
        return top

    def __str__(self) -> str:
        return (
            f"{self.board[0]} {self.board[1]} {self.board[2]} {self.board[3]}\n"
            f"{self.board[4]} {self.board[5]} {self.board[6]} {self.board[7]}\n"
            f"{self.board[8]} {self.board[9]} {self.board[10]} {self.board[11]}\n"
            f"{self.board[12]} {self.board[13]} {self.board[14]} {self.board[15]}"
        )


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
            return time_step.termination(
                np.array(self.board.grid(), dtype=np.int32), reward=0
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

    def render(self, mode="human") -> None:
        print(self.board.pretty())


class LearningAgent:
    env: Union[TFPyEnvironment, None] = None
    eval_env: Union[TFPyEnvironment, None] = None
    global_step: Union[tf.Tensor, None] = None
    agent: Union[DqnAgent, None] = None
    replay_buffer: Union[TFUniformReplayBuffer, None] = None

    def __init__(
        self,
        file: Path,
        reward_function: Callable[[Board], np.float32] = generic_reward,
    ) -> None:
        self.env = TFPyEnvironment(
            GameEnvironment(reward_func=reward_function)
        )
        self.eval_env = TFPyEnvironment(
            GameEnvironment(reward_func=reward_function)
        )
        net: QNetwork = QNetwork(
            input_tensor_spec=self.env.observation_spec(),
            action_spec=self.env.action_spec(),
            fc_layer_params=(128, 128),
        )
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.agent = DqnAgent(
            self.env.time_step_spec(),
            self.env.action_spec(),
            q_network=net,
            optimizer=Adam(learning_rate=1e-3),
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.global_step,
        )
        self.agent.initialize()
        self.replay_buffer: TFUniformReplayBuffer = TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.env.batch_size,
            max_length=100000,
        )
        self.train_checkpointer = common.Checkpointer(
            ckpt_dir=file,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step,
        )
        self.train_checkpointer.initialize_or_restore()

    def no_data(self, dataset) -> bool:
        is_empty = True
        for _ in dataset:
            is_empty = True
            break
        return is_empty

    def train(self, epoch: int) -> None:
        for i in range(epoch):
            time_step = self.env.reset()
            while not time_step.is_last():
                action_step = self.agent.collect_policy.action(time_step)
                next_time_step = self.env.step(action_step.action)
                traj = from_transition(time_step, action_step, next_time_step)
                self.replay_buffer.add_batch(traj)
                time_step = next_time_step
            experience = self.replay_buffer.as_dataset(
                sample_batch_size=16,
                num_steps=2,
                single_deterministic_pass=False,
            )
            try:
                for element in experience.take(1):
                    loss = self.agent.train(element[0])
            except:
                pass

    def run_game(self, render=True) -> np.int32:
        time_step = self.eval_env.reset()
        while not time_step.is_last():
            time_step = self.eval_env.step(
                self.agent.policy.action(time_step).action
            )
            if render:
                self.eval_env.render(mode="human")
        board = Board()
        board.board = time_step[3].numpy().flatten()
        return board.score()

    def save(self) -> None:
        self.train_checkpointer.save(self.global_step)


def interpret_results(results: np.ndarray[np.float64]) -> str:
    output = (
        f"**********RESULTS**********\n * sample size = {len(results)}\n *"
        f" mean = {results.mean()}\n * variance = {np.var(results)}\n * min ="
        f" {results.min()}\n * 25th percentile ="
        f" {np.percentile(results, 25)}\n * median ="
        f" {np.percentile(results, 50)}\n * 75th percentile ="
        f" {np.percentile(results, 75)}\n * max ="
        f" {results.max()}\n***************************"
    )
    return output


if __name__ == "__main__":
    less_tiles_agent = LearningAgent(
        Path(__file__).parent.parent.joinpath("saves").joinpath("less_tiles"),
        reward_function=less_tiles_reward,
    )
    generic_agent = LearningAgent(
        Path(__file__).parent.parent.joinpath("saves").joinpath("generic"),
        reward_function=generic_reward,
    )
    hankel_agent = LearningAgent(
        Path(__file__).parent.parent.joinpath("saves").joinpath("hankel"),
        reward_function=hankel_reward,
    )

    num_games = 100
    less_tiles_scores = np.zeros(num_games, dtype=np.int32)
    generic_scores = np.zeros(num_games, dtype=np.int32)
    hankel_scores = np.zeros(num_games, dtype=np.int32)
    for i in range(num_games):
        less_tiles_scores[i] = less_tiles_agent.run_game(render=False)
        generic_scores[i] = generic_agent.run_game(render=False)
        hankel_scores[i] = hankel_agent.run_game(render=False)
    print(f"LESS TILES\n{interpret_results(less_tiles_scores)}")
    print(f"GENERIC\n{interpret_results(generic_scores)}")
    print(f"HANKEL\n{interpret_results(hankel_scores)}")
