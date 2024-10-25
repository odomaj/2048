import game_logic


def get_move() -> game_logic.Move:
    inp = input("enter move")


if __name__ == "__main__":
    board: game_logic.Board = game_logic.Board()
    while True:
        print(f"{board.score()}\n{board}")
        move = get_move()
        result = board.make_move(move)
        if result == game_logic.MoveResult.LOST:
            break
        elif result == game_logic.MoveResult.WON:
            break
