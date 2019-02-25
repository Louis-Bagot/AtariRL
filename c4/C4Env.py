from game_original import BoardC4
import numpy as np

class C4Env():
    """Gym-like environment for connect 4"""
    def __init__(self):
        self.board = BoardC4()
        self.n_actions = self.board.width
        self.done = False
        self.r = None

    def render(self):
        print(self.board.dico2array())

    def reset(self):
        self.board = BoardC4()
        return self.board.fields2channels()

    def step(self, action):
        self.board.move(action)
        self.done = bool(self.board.won())
        self.r = self.board.opponent if self.done else 0 # already switched
        if not self.done and not any(self.legal_moves()):
            self.done = True
            self.r = 0
        return self.done

    def random_move(self):
        return np.random.choice(self.board.legal_moves())

    def legal_moves(self):
        return self.board.legal_moves()
