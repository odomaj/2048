import tensorflow as tf
from tf_keras.optimizers import Adam
from tf_agents.networks.q_network import QNetwork
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import (
    TFUniformReplayBuffer,
)
from tf_agents.trajectories import from_transition
from tf_agents.utils import common
from typing import Union, Callable
import env.game_environment as ge
from env.game_logic import Board
from pathlib import Path
import numpy as np


class LearningAgent:
    env: Union[TFPyEnvironment, None] = None
    eval_env: Union[TFPyEnvironment, None] = None
    global_step: Union[tf.Tensor, None] = None
    agent: Union[DqnAgent, None] = None
    replay_buffer: Union[TFUniformReplayBuffer, None] = None

    def __init__(
        self,
        file: Path,
        reward_function: Callable[[Board], np.float32] = ge.generic_reward,
    ) -> None:
        self.env = TFPyEnvironment(
            ge.GameEnvironment(reward_func=reward_function)
        )
        self.eval_env = TFPyEnvironment(
            ge.GameEnvironment(reward_func=reward_function)
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
