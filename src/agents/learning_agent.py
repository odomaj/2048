from env.game_environment import GameEnvironment
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
from typing import Union


class LearningAgent:
    env: Union[TFPyEnvironment, None] = None
    agent: Union[DqnAgent, None] = None

    def __init__(self) -> None:
        self.env = TFPyEnvironment(GameEnvironment())
        self.eval_env = TFPyEnvironment(GameEnvironment())
        net: QNetwork = QNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            (128, 128),
        )
        self.agent = DqnAgent(
            self.env.time_step_spec(),
            self.env.action_spec(),
            q_network=net,
            optimizer=Adam(learning_rate=1e-3),
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=tf.Variable(0),
        )
        self.agent.initialize()

    def train(self, epoch: int) -> None:
        replay_buffer: TFUniformReplayBuffer = TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.env.batch_size,
            max_length=100000,
        )
        for _ in range(epoch):
            time_step = self.env.reset()
            while not time_step.is_last():
                action_step = self.agent.collect_policy.action(time_step)
                next_time_step = self.env.step(action_step.action)
                traj = from_transition(time_step, action_step, next_time_step)
                replay_buffer.add_batch(traj)
                time_step = next_time_step
            experience = replay_buffer.as_dataset(
                sample_batch_size=64, num_steps=2
            ).take(1)[0]
            self.agent.train(experience)
