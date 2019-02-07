import numpy as np;

def decay_epsilon(frame, min_decay, no_decay_threshold):
    """Linear decay from 1 (frame 0) to min_decay (frame no_decay_threshold),
       min_decay thereafter"""
    return min_decay if (frame > no_decay_threshold)\
                     else (min_decay-1)*frame/no_decay_threshold +1

def greedy(dqn, obs) :
  return np.argmax(dqn.predict(np.array([obs])))

def eps_greedy(epsilon, n_actions, dqn, obs):
    return np.random.randint(n_actions) if (np.random.rand(1) < epsilon)\
        else greedy(dqn, obs)

def train_dqn(dqn, old_dqn, states, actions, rewards, new_states, gamma):
    # record predictions q(s,a) so only the performed action is modified
    q_targets = dqn.predict(states)
    # compute the max over rows of q(s',.; theta-) ie old values of dqn
    new_q_values = old_dqn.predict(new_states)
    max_new_q = np.max(new_q_values, axis=1)
    # the q_target is the same for any non performed action
    # r + gamma*maxq for performed the action in state
    for i, action in enumerate(actions):
        # reach cell q(s,a) of performed action in state s
        q_targets[i][action] = rewards[i] + gamma*max_new_q[i]

    dqn.train_on_batch(states, q_targets)
