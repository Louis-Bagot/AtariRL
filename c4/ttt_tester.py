from TicTacEnv import TicTacEnv
from TicTacEnv import TicTacEnv
from Random_Agent import Random_Agent
from MCTS import MCTS_Agent
import time
import numpy as np

if __name__ == "__main__":
    agent1 = MCTS_Agent(rollouts=1000)
    agent2 = MCTS_Agent(rollouts=1)
    nb_games = 10
    env = TicTacEnv()

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
                env.step(m)
            res = env.r
            res_rec.append(res)
        print(res_rec)
        print(np.mean(res_rec))
