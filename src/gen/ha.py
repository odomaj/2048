import sys

sys.path.append("..")

from agents.learning_agent import LearningAgent
from pathlib import Path
import env.game_environment as ge
import tensorflow as tf
import numpy as np


def write_output(file: Path, output: str) -> None:
    with file.open("a") as f:
        f.write(f"{output}\n")


def main() -> None:
    # Initialize the learning agent
    hankel_agent = LearningAgent(
        Path(__file__)
        .parent.parent.parent.joinpath("saves")
        .joinpath("hankel"),
        reward_function=ge.hankel_reward,
    )
    num_epochs_per = 400
    num_games_run = 50
    i = 1
    while i:
        hankel_agent.train(num_epochs_per)
        print(
            f"******FINISHED TRAINING ROUND {i} OF"
            f" {num_epochs_per} EPOCHS******"
        )
        hankel_agent.save()
        print("******SAVED AGENTS******")
        ha = np.zeros(num_games_run)
        for j in range(num_games_run):
            ha[j] = hankel_agent.run_game(render=False)
        write_output(
            Path(__file__)
            .parent.parent.parent.joinpath("output")
            .joinpath("hankel_agent.txt"),
            ha,
        )
        i += 1


if __name__ == "__main__":
    main()
