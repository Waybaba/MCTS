"""
An example implementation of the abstract Node class for use in MCTS
If you run this file then you can play against the computer.
A tic-tac-toe board is represented as a tuple of 9 values, each either None,
True, or False, respectively meaning 'empty', 'X', and 'O'.
The board is indexed by row:
0 1 2
3 4 5
6 7 8
For example, this game board
O - X
O X -
X - -
corrresponds to this tuple:
(False, None, True, False, True, None, True, None, None)
"""

from collections import namedtuple
from random import choice
from monte_carlo_tree_search import MCTS, Node
import gym

_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")

_TTTTB = namedtuple("FrozenLakeBoard", "env history terminal reward_")



# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class TicTacToeBoard(_TTTB, Node):
    def find_children(board):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {
            board.make_move(i) for i, value in enumerate(board.tup) if value is None # tup {number: True/None} None means this is not choosed yet
        }

    def find_random_child(board):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        empty_spots = [i for i, value in enumerate(board.tup) if value is None]
        return board.make_move(choice(empty_spots))

    def reward(board):
        if not board.terminal:
            raise RuntimeError(f"reward called on nonterminal board {board}")
        if board.winner is board.turn:
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError(f"reward called on unreachable board {board}")
        if board.turn is (not board.winner):
            return 0  # Your opponent has just won. Bad.
        if board.winner is None:
            return 0.5  # Board is a tie
        # The winner is neither True, False, nor None
        raise RuntimeError(f"board has unknown winner type {board.winner}")

    def is_terminal(board):
        return board.terminal

    def make_move(board, index):
        tup = board.tup[:index] + (board.turn,) + board.tup[index + 1 :]
        turn = not board.turn
        winner = _find_winner(tup)
        is_terminal = (winner is not None) or not any(v is None for v in tup)
        return TicTacToeBoard(tup, turn, winner, is_terminal)

    def to_pretty_string(board):
        to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [
            [to_char(board.tup[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )


class FrozenLakeBoard(_TTTTB, Node):

    def find_children(board):
        if board.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {
            board.make_move(a) for a in FrozenLakeEnv.actions
        }

    def find_random_child(board):
        if board.terminal:
            return None  # If the game is finished then no moves can be made
        return board.make_move(choice(FrozenLakeEnv.actions))

    def reward(board):
        if not board.terminal or (board.reward is None):
            raise RuntimeError(f"reward called on nonterminal board {board}")
        return board.reward_

    def is_terminal(board):
        return board.terminal

    def make_move(board, index):
        # tup = board.tup[:index] + (board.turn,) + board.tup[index + 1 :]
        # turn = not board.turn
        # winner = _find_winner(tup)
        is_terminal, reward = board.env.make_move(index)
        new_history = board.history + str(index)
        env = FrozenLakeEnv(setting=board.env.setting, history=new_history) # new back env, it will move to that step
        return FrozenLakeBoard(env, new_history, is_terminal, reward)

    def to_pretty_string(board):
        board.env.render()
        return "Finish render"

class ModifiedFrozenLakeGymEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.t = 0.
    
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        # modify ...
        s_type = self.env.desc.flatten()[next_state]
        self.t += 0.1
        if s_type == b'F': reward = 0.
        elif s_type == b'H': reward = -0.01
        elif s_type == b'G': reward = max(1, (+10. - self.t))

        return next_state, reward, done, info

class FrozenLakeEnv:
    actions = [0, 1, 2, 3]

    def __init__(self, setting, history):
        self.setting = setting
        self.history = history
    
        self.env = ModifiedFrozenLakeGymEnv(gym.make(setting['name'], is_slippery=setting['is_slippery']))
        self.reset_to(history) # init with this {'name': 'FrozenLake-v0', 'is_slippery': False}
        return
    
    def reset_to(self, history):
        self.env.reset()
        for a in history:
            self.env.step(int(a))
        return
    
    def make_move(self, index):
        "only return is_terminal and reward, without making real move"
        new_env = self.copy()
        s, reward, is_terminal, info = new_env.env.step(index)
        # cp env and return is_terminal
        # if terminal return reward, else reward = None
        return is_terminal, reward if is_terminal else None
    
    def render(self):
        self.env.render()
    
    def copy(self):
        return FrozenLakeEnv(self.setting, self.history)



def play_game():
    tree = MCTS()
    # board = new_tic_tac_toe_board()
    board = new_frozen_lake_board()
    print(board.to_pretty_string())
    while True:
        # row_col = input("enter row,col: ")
        # row, col = map(int, row_col.split(","))
        # index = 3 * (row - 1) + (col - 1)
        # if board.tup[index] is not None:
        #     raise RuntimeError("Invalid move")
        # board = board.make_move(index)
        # print(board.to_pretty_string())
        # if board.terminal:
        #     break
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in range(1000):
            tree.do_rollout(board)
        board = tree.choose(board)
        print(board.to_pretty_string())
        if board.terminal:
            break


def _winning_combos():
    for start in range(0, 9, 3):  # three in a row
        yield (start, start + 1, start + 2)
    for start in range(3):  # three in a column
        yield (start, start + 3, start + 6)
    yield (0, 4, 8)  # down-right diagonal
    yield (2, 4, 6)  # down-left diagonal


def _find_winner(tup):
    "Returns None if no winner, True if X wins, False if O wins"
    for i1, i2, i3 in _winning_combos():
        v1, v2, v3 = tup[i1], tup[i2], tup[i3]
        if False is v1 is v2 is v3:
            return False
        if True is v1 is v2 is v3:
            return True
    return None


def new_tic_tac_toe_board():
    return TicTacToeBoard(tup=(None,) * 9, turn=True, winner=None, terminal=False)

def new_frozen_lake_board():
    setting = {'name': 'FrozenLake-v0', 'is_slippery': False}
    history = ""
    return FrozenLakeBoard(env=FrozenLakeEnv(setting, history), history=history, terminal=False, reward_=0.)

if __name__ == "__main__":
    play_game()