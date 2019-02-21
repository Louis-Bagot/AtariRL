# coding=UTF8
from copy import copy, deepcopy
from time import time
import numpy as np

class TicTacEnv:
    def __init__(self,other=None):
        self.player = 1
        self.opponent = -1
        self.empty = 0
        self.trans = {self.player:[1,0], self.opponent:[0,1], self.empty:[0,0]}
        self.width = 3
        self.height = 3
        self.dim = self.height * self.width
        self.fields = np.zeros(self.dim, np.int8)

    def switch_player(self):
        self.player, self.opponent = self.opponent, self.player

    def move(self,move):
        if (move in self.legal_moves()):
            self.fields[move] = self.player
            self.switch_player()
        else:
            raise ValueError("Not legal move : ", x)

    def legal_moves(self):
        return np.flatnonzero(self.fields == self.empty)

    def random_move(self):
        return np.random.choice(self.legal_moves())

    def render(self):
        print(self.fields.reshape((self.height, self.width)))

    def reset(self):
        self.fields = np.zeros(self.dim, np.int8)
        return self.fields

    def copy_other(self, other):
        self.fields = other.fields.copy()

    def make_copy(self):
        c = TicTacEnv()
        c.fields = self.fields.copy()
        return c

    def getFields(self):
        return self.fields.copy()

    def won(self):
        arr = self.fields.reshape((self.height, self.width))
        if any(np.sum(arr, axis=0)==3*self.opponent):
            return True
        if any(np.sum(arr, axis=1)==3*self.opponent):
            return True
        if np.trace(arr)==3*self.opponent:
            return True
        if ((arr[0,2]+arr[1,1]+arr[2,0])==3*self.opponent):
            return True
        return False

    def step(self, action):
        self.move(action)
        o = self.fields
        d = self.won()
        r = self.opponent if d else 0 # already switched
        if not any(self.legal_moves()) and not d:
            d = True
            r = 0
        i = ""
        return o, r, d, i
