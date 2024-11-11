import sys

sys.path.append("..")
import tensorflow as tf
import numpy as np
from typing import Union, Callable
from env.game_logic import Board, Move, MoveResult
from random import randint
from pathlib import Path

EARLY_GAME_SIZE = 10000
LATE_GAME_SIZE = 10000
BATCH_SIZE = 100
EPOCHS = 100


class BasicAgent:
    model: Union[tf.keras.models.Sequential, None] = None

    def fit(
        self,
        dataset: tf.data.Dataset,
        batch_size: np.int32,
        epochs: np.int32,
    ) -> None:
        if model is None:
            return
        self.model.fit(dataset.batch(batch_size), epochs=epochs)

    def predict(self, data: np.ndarray[np.int32]):
        if model is None:
            return
        if model is None:
            return
        return self.model.predict(data)

    def new(self) -> None:
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(16,)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(4, activation="sigmoid"),
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def save(self, file: str) -> None:
        if model is None:
            return
        self.model.save(Path(__file__).parent.joinpath(file))

    def load(self, file: str) -> None:
        self.model = tf.keras.models.load_model(
            Path(__file__).parent.joinpath(file)
        )


def create_dataset(
    boards: np.ndarray[np.ndarray[np.int32]],
    moves: np.ndarray[np.int32],
) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices((boards, moves))


def gen_rand_tile(dist: np.ndarray[np.int32]) -> np.int32:
    return dist[randint(0, len(dist) - 1)]


def best_move(
    board: Board, reward: Callable[[Board, Move], MoveResult]
) -> Move:
    moves = np.zeros(4)
    for i in range(len(moves)):
        moves[i] = reward(board, Move(i + 1))
    return Move(moves.argmax() + 1)


def gen_rand_board(
    tile_dist: np.ndarray[np.int32],
) -> np.ndarray[np.int32]:
    board: Board = Board()
    for i in range(len(board.board)):
        board.board[i] = gen_rand_tile(tile_dist)
    return np.array(board.board, dtype=np.int32)


def normal_dist() -> np.ndarray[np.int32]:
    return np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def exp_dist() -> np.ndarray[np.int32]:
    dist = np.zeros(2047, dtype=np.int32)
    # 1024 0s
    for i in range(0, 1024):
        dist[i] = 0
    # 512 2s
    for i in range(1024, 1536):
        dist[i] = 1
    # 256 4s
    for i in range(1536, 1792):
        dist[i] = 2
    # 128 8s
    for i in range(1792, 1920):
        dist[i] = 3
    # 64 16s
    for i in range(1920, 1984):
        dist[i] = 4
    # 32 32s
    for i in range(1984, 2016):
        dist[i] = 5
    # 16 64s
    for i in range(2016, 2032):
        dist[i] = 6
    # 8 128s
    for i in range(2032, 2040):
        dist[i] = 7
    # 4 256s
    for i in range(2040, 2044):
        dist[i] = 8
    # 2 512s
    for i in range(2044, 2046):
        dist[i] = 9
    # 1 1024
    dist[2046] = 10
    return dist


def hankel_reward(board: np.ndarray[np.int32], move: Move) -> np.float32:
    """Compute the reward based on the Hankel matrix"""
    game_board: Board = Board()
    game_board.board = np.array(board, dtype=np.float64)
    result = game_board.make_move(move)
    if result == MoveResult.ILLEGAL_MOVE or result == MoveResult.LOST:
        return np.finfo(np.float32).min
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
    return np.dot(game_board.board, hankel_matrix)


if __name__ == "__main__":
    early_game: np.ndarray[np.int32] = exp_dist()
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
    model: BasicAgent = BasicAgent()
    model.new()
    model.fit(training_data, EARLY_GAME_SIZE + LATE_GAME_SIZE, EPOCHS)
    model.save("model.keras")
