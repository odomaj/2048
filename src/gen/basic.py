import sys

sys.path.append("..")

from agents.basic_agent import (
    BasicAgent,
    create_dataset,
    gen_rand_board,
    exp_dist,
    normal_dist,
    best_move,
    hankel_reward,
)
from pathlib import Path
import env.game_environment as ge
from typing import Union
import tensorflow as tf
import numpy as np


def write_output(file: Path, output: str) -> None:
    with file.open("a") as f:
        f.write(f"{output}\n")


def main() -> None:
    EARLY_GAME_SIZE = 20000
    LATE_GAME_SIZE = 20000
    TEST_GAMES = 200
    EPOCHS = 500
    early_game: np.ndarray[np.int32] = exp_dist()
    late_game: np.ndarray[np.int32] = normal_dist()
    late_game: np.ndarray[np.int32] = normal_dist()
    boards: list[Union[np.ndarray[np.int32], None]] = [None] * (
        EARLY_GAME_SIZE + LATE_GAME_SIZE
    )
    moves: list[Union[np.int32, None]] = [None] * (
        EARLY_GAME_SIZE + LATE_GAME_SIZE
    )
    for i in range(0, EARLY_GAME_SIZE):
        boards[i] = gen_rand_board(early_game)
        moves[i] = np.int32(int(best_move(boards[i], hankel_reward).value) - 1)
    for i in range(EARLY_GAME_SIZE, EARLY_GAME_SIZE + LATE_GAME_SIZE):
        boards[i] = gen_rand_board(late_game)
        moves[i] = np.int32(int(best_move(boards[i], hankel_reward).value) - 1)

    training_data: tf.data.Dataset = create_dataset(boards, moves)
    basic_agent = BasicAgent()
    basic_agent.new()
    basic_agent.predict(np.zeros(16, dtype=np.int32))
    basic_agent.fit(training_data, EARLY_GAME_SIZE + LATE_GAME_SIZE, EPOCHS)
    basic_agent.save(
        Path(__file__)
        .parent.parent.parent.joinpath("saves")
        .joinpath("basic_agent.keras")
    )

    basic_results = np.zeros(TEST_GAMES, dtype=np.int32)
    for i in range(TEST_GAMES):
        basic_results[i] = basic_agent.run_game()

    write_output(
        Path(__file__)
        .parent.parent.parent.joinpath("output")
        .joinpath("basic_agent.txt"),
        basic_results,
    )


if __name__ == "__main__":
    main()
