import numpy as np
from enum import Enum
from random import randint
from typing import Union


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
        self.board[i] = 1
        return True

    def full_board(self) -> bool:
        """True if there are no 0 tiles on the board, otherwise False"""
        for tile in self.board:
            if tile == 0:
                return False
        return True

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
