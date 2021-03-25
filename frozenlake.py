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

_TTTTB = namedtuple("FrozenLakeBoard", "env history terminal reward_")


class FrozenLakeBoard(_TTTTB, Node):
    """A FrozenLakeBoard which is similar to to TicTacToeBoard
    """
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
    """Environment Wrapper

    Wrap the environment to do reward shaping, stop the env after x steps.

    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.t = 0.
    
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        # modify ...
        s_type = self.env.desc.flatten()[next_state]
        self.t += 1.
        if s_type == b'F': reward = 0.
        elif s_type == b'H': reward = -0.01
        elif s_type == b'G': reward = max(1, (+10. - 0.1*self.t))
        if self.t > 100: done = True

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
        "reset the agent to some position according to the history"
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
        for _ in range(500):
            tree.do_rollout(board)
        board = tree.choose(board)
        print(board.to_pretty_string())
        if board.terminal:
            break

def new_frozen_lake_board():
    setting = {'name': 'FrozenLake-v0', 'is_slippery': False}
    history = ""
    return FrozenLakeBoard(env=FrozenLakeEnv(setting, history), history=history, terminal=False, reward_=0.)

if __name__ == "__main__":
    play_game()