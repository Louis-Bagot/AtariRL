import numpy as np
import tensorflow as tf
import gym
import time
import random

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def generate_input_from_index(i, replay_memory, agent_history_length):
    """Generates a DQN-predictable input from index i of replay_memory"""
    last_frames = replay_memory[(i-agent_history_length+1):(i+1)] # isolate seq
    states = np.array(last_frames)[:,0]/255 # isolate states and normalize
    return np.stack(states)

def extract_mini_batch(replay_memory, batch_size, agent_history_length):
    # first randomly select the indexes
    batch_indexes = np.random.choice(range(agent_history_length,\
                                           len(replay_memory)-1), \
                                     batch_size, replace=False)
    mini_batch = []
    for i in range(batch_size):
        index = batch_indexes[i] # we'll be needing thismany times
        state = generate_input_from_index(index, replay_memory, \
                                          agent_history_length)
        new_state = generate_input_from_index(index +1, replay_memory,\
                                              agent_history_length)
        # (observation, action, reward, done) and add new_s in 3rd pos
        mini_batch.append((state, \
                          replay_memory[index][1], \
                          replay_memory[index][2], \
                          new_state, \
                          replay_memory[index][3]))

    return np.array(mini_batch)

def decay_epsilon(frame, min_decay, no_decay_threshold):
    return min_decay if (frame > no_decay_threshold)\
                     else (min_decay-1)*frame/no_decay_threshold +1

def greedy(dqn, frame_seq, new_algo, n_actions):
    return np.argmax(dqn.predict(np.array([frame_seq]))) if not new_algo else\
           np.argmax(dqn.predict([np.array([frame_seq]), np.array([np.ones(n_actions)])]))

def random_action(n_actions):
    return np.random.randint(n_actions)

def eps_greedy(epsilon, n_actions, dqn, replay_memory, agent_history_length, new_algo):
    return random_action(n_actions) if (np.random.rand(1) < epsilon)\
        else greedy(dqn, generate_input_from_index(len(replay_memory)-1, \
                                                   replay_memory, \
                                                   agent_history_length), \
                    new_algo, n_actions)

def copy_model(model):
    """Returns a copy of a keras model."""
    model.save('tmp_model')
    return tf.keras.models.load_model('tmp_model')

def train_dqn(dqn, old_dqn, mini_batch, gamma):
    # extract s a r s' from mini_batch; and the terminal states
    states =  np.stack(mini_batch[:,0], axis=0)
    actions = mini_batch[:,1]
    rewards = mini_batch[:,2]
    new_states =  np.stack(mini_batch[:,3], axis=0)
    dones = np.array(mini_batch[:,4])
    # record predictions q(s,a) so only the performed action will be modified
    q_targets = dqn.predict(states)
    # compute the max over rows of q(s',.; theta-) ie old values of dqn
    new_q_values = old_dqn.predict(new_states)
    max_new_q = np.max(new_q_values, axis=1)
    # the q_target is the same for any non performed action
    # r + gamma*maxq for the performed action in state s'
    for i, action in enumerate(actions):
        # reach cell q(s,a) of performed action in state s.
        q_targets[i][action] = rewards[i] + (1-dones[i])*gamma*max_new_q[i]

    dqn.train_on_batch(states, q_targets)

def train_dqn2(dqn, old_dqn, mini_batch, gamma, n_actions):
    # extract s a r s' from mini_batch; and the terminal states
    states =  np.stack(mini_batch[:,0], axis=0)
    actions = one_hot(mini_batch[:,1].astype(np.int8), n_actions)
    rewards = mini_batch[:,2]
    new_states =  np.stack(mini_batch[:,3], axis=0)
    dones = np.array(mini_batch[:,4]).astype(np.bool)
    # compute the max over rows of q(s',.; theta-) ie old values of dqn
    new_q_values = old_dqn.predict([new_states, np.ones(actions.shape)])
    new_q_values[dones] = 0
    # q update: reward + gamma * max new state q
    q_targets = rewards + gamma * np.max(new_q_values, axis=1)
    dqn.fit([states, actions], actions * q_targets[:, None],\
            epochs=1, batch_size=len(states), verbose=0)

def test_dqn(game, test_explo, dqn, agent_history_length, new_algo):
    print("\tEntering test phase...")
    env = gym.make(game) # environment
    n_actions = env.action_space.n
    replay_memory = []
    max_memory = agent_history_length
    max_episode = 30
    score_record = [] # episode scores over time (episodes)
    frame = 0
    ## Test loop
    for i_episode in range(0,max_episode):
        # init observation
        observation = preprocess(env.reset())
        done = False
        cumul_score = 0 #episode total score
        #Game loop
        while not done:
            if (frame > agent_history_length):
                action = eps_greedy(test_explo, n_actions, dqn,replay_memory,\
                                    agent_history_length, new_algo)
            else : action = random_action(n_actions)

            observation, reward, done, info = env.step(action)
            observation = preprocess(observation)
            reward = np.sign(reward)
            cumul_score += reward
            frame += 1

            # replay memory handling
            replay_memory.append((observation, action, reward, done))
            if len(replay_memory) > max_memory:
                replay_memory.pop(0)

        score_record.append(cumul_score)

    print("\tScores on ", max_episode, " episodes : ", score_record)
    avg_score = np.mean(score_record)
    print("\tReturning ", avg_score)
    return avg_score

def keep_playing(game, test_explo, dqn, agent_history_length, new_algo):
    print("Now showing off them mad skillz")
    env = gym.make(game) # environment
    n_actions = env.action_space.n
    replay_memory = []
    max_memory = agent_history_length
    score_record = [] # episode scores over time (episodes)
    frame = 0
    ## Play forever
    while True:
        # init observation
        observation = preprocess(env.reset())
        done = False
        cumul_score = 0 #episode total score
        #Game loop
        while not done:
            if (frame > agent_history_length):
                action = eps_greedy(test_explo, n_actions, dqn, replay_memory,\
                                    agent_history_length, new_algo)
            else : action = random_action(n_actions)
            env.render()
            time.sleep(.05)
            observation, reward, done, info = env.step(action)
            observation = preprocess(observation)
            reward = np.sign(reward)
            cumul_score += reward
            frame += 1

            # replay memory handling
            replay_memory.append((observation, action, reward, done))
            if len(replay_memory) > max_memory:
                replay_memory.pop(0)

        print("\tScore on this episode : ", cumul_score)
