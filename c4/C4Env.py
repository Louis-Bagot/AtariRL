from game_original import Board
import numpy as np

class C4Env():
    """Gym-like environment for connect 4"""
    def __init__(self):
        self.board = Board()
        self.n_actions = self.board.width

    def render(self):
        print(self.board.dico2array())

    def reset(self):
        self.board = Board()
        return self.board.fields2channels()

    def step(self, action):
        self.board.move(action)
        o = self.board.fields2channels()
        d = bool(self.board.won())
        r = self.board.opponent if d else 0 # already switched
        if not any(self.legal_moves()):
            d = True
            r = 0
        i = ""
        return o, r, d, i

    def random_move(self):
        moves = np.array(range(self.board.width))
        return np.random.choice(moves[self.board.legal_moves()])

    def legal_moves(self):
        return self.board.legal_moves()
