from game_original import Board
import numpy as np

class C4Env():
    """Gym-like environment for connect 4"""
    def __init__(self):
        self.board = Board()
        self.n_actions = self.board.width

    def dico2array(self, dico, height, width):
        return np.array([[dico[x,y] for x in range(width)] \
                                    for y in range(height)])

    def render(self):
        print(dico2array(self.board.fields, \
                         self.board.height, self.board.width))

    def reset(self):
        self.board = Board()
        return self.dico2array(self.board.fields, \
                               self.board.height, self.board.width)

    def step(self, action):
        self.board.move(action)
        o = self.dico2array(self.board.fields, \
                       self.board.height, self.board.width)
        d = bool(self.board.won())
        if d:
            r = self.board.opponent # since we already switched in move
        else :
            r = 0
        if not any(self.board.legal_moves()):
            d = True
            r = 0
        i = ""
        return o, r, d, i

    def random_move(self):
        moves = np.array(range(self.board.width))
        return np.random.choice(moves[self.board.legal_moves()])
