import tensorflow as tf
import numpy as np
from typing import Union, Callable
from random import randint
from pathlib import Path
from enum import Enum

EARLY_GAME_SIZE = 10000
LATE_GAME_SIZE = 10000
BATCH_SIZE = 100
EPOCHS = 100


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

    def predict(
        self, data: np.ndarray[np.ndarray[np.int32]]
    ) -> np.ndarray[np.float32]:
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
    model.fit(training_data, BATCH_SIZE, EPOCHS)

    test: np.ndarray[np.ndarray[np.int32]] = np.array(
        [gen_rand_board(early_game)]
    )
    print(model.predict(test))

    model.save("model.keras")
