import game_logic


def get_move() -> game_logic.Move:
    while True:
        inp = input("enter move\n")
        if inp == "w":
            return game_logic.Move.UP
        elif inp == "s":
            return game_logic.Move.DOWN
        elif inp == "a":
            return game_logic.Move.LEFT
        elif inp == "d":
            return game_logic.Move.RIGHT


if __name__ == "__main__":
    board: game_logic.Board = game_logic.Board()
    while True:
        print(f"{board.score()}\n{board.pretty()}")
        move = get_move()
        result = board.make_move(move)
        if result == game_logic.MoveResult.LOST:
            break
        elif result == game_logic.MoveResult.WON:
            break
