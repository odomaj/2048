import tensorflow as tf
import numpy as np
from typing import Union
from env.game_logic import Board, Move, MoveResult
from random import randint


class BasicAgent:
    model: Union[tf.keras.models.Sequential, None] = None

    def __init__(self) -> None:
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

    def fit(
        self,
        dataset: tf.data.Dataset,
        batch_size: np.int32,
        epochs: np.int32,
    ) -> None:
        self.model.fit(dataset.batch(batch_size), epochs)

    def predict(self):
        pass


def create_dataset(
    boards: np.ndarray[np.ndarray[np.int32]],
    moves: np.ndarray[np.int32],
) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices((boards, moves))


def gen_rand_tile(dist: np.ndarray[np.int32]) -> np.int32:
    return dist[randint(0, len(dist) - 1)]


def best_move(board: Board, reward: function) -> Move:
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
    return (np.array(board.board, dtype=np.int32),)


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


def hankel_reward(board: Board, move: Move) -> np.float32:
    """Compute the reward based on the Hankel matrix"""
    board_save = np.array(board.board)
    result = board.make_move(move)
    new_board = np.array(board.board)
    board.board = board_save
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
    return np.dot(new_board, hankel_matrix)


if __name__ == "__main__":
    early_game: np.ndarray[np.int32] = exp_dist()
    late_game: np.ndarray[np.int32] = normal_dist()
    boards: list[Union[np.ndarray[np.int32], None]] = [None] * 2000
    moves: list[Union[np.int32, None]] = [None] * 2000
    for i in range(0, 1000):
        boards[i] = gen_rand_board(early_game)
        moves[i] = np.int32(int(best_move(boards[i], hankel_reward)) - 1)
    for i in range(1000, 2000):
        boards[i] = gen_rand_board(late_game)
        moves[i] = np.int32(int(best_move(boards[i], hankel_reward)) - 1)

    training_data: tf.data.Dataset = create_dataset(
        np.array(boards), np.array(moves)
    )

    model: BasicAgent = BasicAgent()
    model.fit(training_data, 100, 10)
