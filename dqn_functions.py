import numpy as np;

def decay(frame, min_decay, no_decay_threshold):
    """Linear decay from 1 (frame 0) to min_decay (frame no_decay_threshold),
       min_decay thereafter"""
    return min_decay if (frame > no_decay_threshold)\
                     else (min_decay-1)*frame/no_decay_threshold +1

def eps_greedy(epsilon, n_actions, dqn, obs):
    return np.random.randint(n_actions) if (np.random.rand(1) < epsilon)\
        else np.argmax(dqn.predict(observation))
