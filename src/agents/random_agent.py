from env.game_logic import Board, Move, MoveResult
from typing import Union
from random import randint


class RandomAgent:
    game: Union[Board, None] = None
    score: Union[int, None] = None
    largest_tile: Union[int, None] = None

    def __init__(self) -> None:
        self.game = Board()

    def get_move(self) -> Move:
        return Move(randint(1, 4))

    def run_game(self) -> bool:
        won = False
        while True:
            result: MoveResult = self.game.make_move(self.get_move())
            if result == MoveResult.WON:
                won = True
                break
            elif result == MoveResult.LOST:
                break
        self.score = int(self.game.score())
        self.largest_tile = int(self.game.largest_tile())
        return won
