
class MCTS_node():
    """ State-action Node of the MC Tree Search.
    """
    def __init__(self, state, move = None, parent = None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.untried_moves = state.legal_moves()
        self.q = 0
        self.n = 0

    def add_child(self, move, state):
        pass


class MCTS_agent():
    """class that handles a MCTS agent"""
    def __init__(self, rollout):
        self.rollout = rollout

    def select(self, node, state):
        
        pass

    def expand(self, node, state):
        pass

    def playout(self, node, state):
        pass

    def backprop(self, node, state):
        pass

    def play(root_state):
        """ Simulate a fixed amount of rollouts.
            A single simulation is a (select, expand, playout, backprop) series.
            """
        root_node = MCTS_node(root_state)
        for simulation in range(rollout):
            node = root_node # will be changed
            state = root_state.make_copy() # shouldn't be changed
            # Perform the chain s,e,p,b
            self.select  (node, state)
            self.expand  (node, state)
            self.playout (node, state)
            self.backprop(node, state)

        # now grab the most visited move - ie child, and play its move
        visit_counts = [child.n for child in root_node.children]
        best_child = np.argmax(visit_counts)
        return best_child.move
