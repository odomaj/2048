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
import numpy as np


class LearningAgent:
    env: Union[TFPyEnvironment, None] = None
    agent: Union[DqnAgent, None] = None

    def __init__(self) -> None:
        self.env = TFPyEnvironment(GameEnvironment())
        self.eval_env = TFPyEnvironment(GameEnvironment())
        net: QNetwork = QNetwork(
            input_tensor_spec=self.env.observation_spec(),
            action_spec=self.env.action_spec(),
            fc_layer_params=(128, 128),
        )
        # print(
        #    np.shape(self.env.time_step_spec()),
        #    np.shape(self.env.action_spec()),
        #    np.shape(self.env.observation_spec()),
        # )
        self.agent = DqnAgent(
            self.env.time_step_spec(),
            self.env.action_spec(),
            q_network=net,
            optimizer=Adam(learning_rate=1e-3),
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=tf.Variable(0).value(),
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
<<<<<<< Updated upstream
                #print(traj)
=======
>>>>>>> Stashed changes
                replay_buffer.add_batch(traj)
                time_step = next_time_step
            experience = replay_buffer.as_dataset(
                sample_batch_size=64,
                num_steps=2,
                single_deterministic_pass=False,
            )
            print(
                "**************************************************************"
            )
            print("                       full dataset")
            print(
                "**************************************************************"
            )
            print(experience)
            print(
                "**************************************************************"
            )
            print("                         take 1")
            print(
                "**************************************************************"
            )
            print(experience.take(1))
            print(
                "**************************************************************"
            )
            print("                         data spec")
            print(
                "**************************************************************"
            )
            print(self.agent.training_data_spec)
            print(
                "**************************************************************"
            )
            print("                         kaboom")
            print(
                "**************************************************************"
            )
            # self.agent.train(self.agent.training_data_spec)
            # print("that worked")
            for element in experience.take(1):
                useful_thing = element[0]
                print(useful_thing)
                self.agent.train(useful_thing)
            print(
                "--------------------------------------------------------------------------------------------------"
            )
            # self.agent.train(experience)
            print("please print")
