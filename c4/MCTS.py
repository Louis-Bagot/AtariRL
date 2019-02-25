# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a
# state.GetRandomMove() or state.DoRandomRollout() function.
#
# Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
# can write your own GameState use UCT in your 2-player game. Change the game to be played in
# the UCTPlayGame() function at the bottom of the code.
#
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
#
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
#
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai

from math import *
import random
import numpy as np
from TicTacEnv import TicTacEnv
class MCTS_Node:
    """ A node in the game tree. Note w is always from the viewpoint of playerJustMoved.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parent = parent # "None" for the root node
        self.children = []
        self.w = 0
        self.n = 0
        self.q = 0
        self.uct_cst = 1
        self.untried_moves = state.legal_moves() # future child nodes
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the MCTS_Node needs later

    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node.
        """
        return sorted(self.children, key = lambda c: c.q + self.uct_cst * sqrt(2*log(self.n)/c.n))[-1]


    def add_child(self, m, s):
        """ Remove m from untried_moves and add a new child node for this move.
            Returns the added child node
        """
        child = MCTS_Node(move = m, parent = self, state = s)
        self.untried_moves.remove(m)
        self.children.append(child)
        return child

    def update(self, result):
        """ Update this node - one additional visit and result additional w. result must be from the viewpoint of playerJustmoved.
        """
        self.n += 1
        self.w += result
        self.q = self.w/self.n

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.w) + "/" + str(self.n) + " U:" + str(self.untried_moves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.children:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.children:
             s += str(c) + "\n"
        return s

class MCTS_Agent(object):
    """ Agent that plays using a MCTS."""
    def __init__(self, rollouts=100):
        self.rollouts = rollouts

    def select(self, node, state):
        while node.untried_moves == [] and node.children != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.step(node.move)
        return node

    def expand(self, node, state):
        if node.untried_moves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untried_moves)
            state.step(m)
            node = node.add_child(m,state) # add child and descend tree
        return node

    def playout(self, node, state):
        """while state.legal_moves() != []: # while state is non-terminal
            state.step(random.choice(state.legal_moves()))
        return state.win_pov(node.playerJustMoved)"""
        return state.playerJustMoved * state.finish_randomly()

    def backprop(self, node, g):
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.update(g) # state is terminal. update node with result from POV of node.playerJustMoved
            g *= -1
            node = node.parent

    def play(self, rootstate, verbose = False):
        """ Conduct a UCT search for itermax iterations starting from rootstate.
            Return the best move from the rootstate.
            Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

        rootnode = MCTS_Node(state = rootstate)
        for i in range(self.rollouts):
            node = rootnode
            state = rootstate.clone()

            node = self.select (node, state)
            node = self.expand (node, state)
            g    = self.playout(node, state)
            self.backprop(node, g)
        # Output some information about the tree - can be omitted
        if (verbose):
            print( rootnode.TreeToString(0))
            print( rootnode.ChildrenToString())

        return sorted(rootnode.children, key = lambda c: c.n)[-1].move # return the move that was most visited
