from C4Env import C4Env
from Alpha4 import Alpha4
from Random_Agent import Random_Agent
from MCTS import MCTS_Agent
import numpy as np
import time

if __name__ == '__main__':
    env.render()
    l = env.legal_moves()
    print(l)
    #agent1 = MCTS_Agent(rollouts=1000)
    #agent1 = MCTS_Agent(rollouts=1000)
    agent1 = Random_Agent()
    agent2 = Random_Agent()
    nb_games = 10
    env = C4Env()

    for snd_player in [-1,1]:
        if snd_player == -1:
            print("MCTS 1000 as White:")
        else:
            print("MCTS 1000 as Black:")
        res_rec=[]
        for ep in range(nb_games):
            env.reset() # uncomment to play Nim with the given number of starting chips
            while (not env.done):
                if env.playerJustMoved == snd_player:
                    m = agent1.play(rootstate = env) # play with values for itermax and verbose = True
                else:
                    m = agent2.play(rootstate = env) # play with values for itermax and verbose = True
                env.move(m)
            res = env.win_pov(1)
            res_rec.append(res)
        print(res_rec)
        print(np.mean(res_rec))

""" a4 = Alpha4(observation.shape, env.n_actions)
    p = a4.predict(np.array([observation]))
    print(p)
    print(p[0]*l)
"""
