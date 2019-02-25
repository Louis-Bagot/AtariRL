from math import *
import random
import numpy as np

class TicTacEnv:
    """ A state of the game, i.e. the game board. players & empty : (1 -1 0)"""
    def __init__(self):
        self.playerJustMoved = -1 # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = [0,0,0,0,0,0,0,0,0] # 0 = empty, 1 = player 1, 2 = player 2
        self.done = False
        self.r = None

    def clone(self):
        """ Create a deep clone of this game state."""
        st = TicTacEnv()
        st.playerJustMoved = self.playerJustMoved
        st.board = self.board[:]
        st.done = self.done
        st.r = self.r
        return st

    def switch_player(self):
        self.playerJustMoved *= -1

    def step(self, move):
        """ update a state by switching players and carrying out the given move.
        """
        assert move in self.legal_moves()
        self.switch_player()
        self.board[move] = self.playerJustMoved
        self.win_check()
        return self.done

    def legal_moves(self):
        """ Get all possible moves from this state."""
        return [] if self.done else \
               [i for i in range(9) if self.board[i] == 0]

    def random_move(self):
        """ Get all possible moves from this state."""
        return random.choice(self.legal_moves())

    def win_check(self):
        """ Get the game result from the viewpoint of playerjm."""
        for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
            if self.board[x] == self.board[y] == self.board[z] != 0:
                self.done = True
                self.r = self.board[x]
                return self.r

        if self.legal_moves() == []:
            self.done = True
            self.r = 0

        return self.r

    def finish_randomly(self):
        """ Returns the winner of a random game from now on"""
        while not self.done:
            self.step(self.random_move())
        return self.r

    def render(self):
        s= ""
        for i in range(9):
            s += ".XO"[self.board[i]]
            if i % 3 == 2: s += "\n"
        print(s)

    def reset(self):
        self.__init__()
