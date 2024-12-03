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
    generic_agent = LearningAgent(
        Path(__file__)
        .parent.parent.parent.joinpath("saves")
        .joinpath("generic"),
        reward_function=ge.generic_reward,
    )
    num_epochs_per = 400
    num_games_run = 50
    i = 1
    while i:
        generic_agent.train(num_epochs_per)
        print(
            f"******FINISHED TRAINING ROUND {i} OF"
            f" {num_epochs_per} EPOCHS******"
        )
        generic_agent.save()
        print("******SAVED AGENTS******")
        ga = np.zeros(num_games_run)
        for j in range(num_games_run):
            ga[j] = generic_agent.run_game(render=False)
        write_output(
            Path(__file__)
            .parent.parent.parent.joinpath("output")
            .joinpath("generic_agent.txt"),
            ga,
        )
        i += 1


if __name__ == "__main__":
    main()
