import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from dqn_functions import *
from wrappers2 import wrap_dqn

def graph(l, xlab, ylab, title, game_name):
    f = plt.figure()
    p = f.add_subplot(111)
    p.plot(l)
    p.set_xlabel(xlab)
    p.set_ylabel(ylab)
    p.set_title(title)
    plt.savefig('../graphic/perf_' + game_name + '.png')

def print_info(frame, episode, epoch_nb, memory_usage, memory_capacity, epsilon):
    print("\nEpoch ", epoch_nb)
    print("\tEpisode ", episode)
    print("\tFrame ", frame)
    print("\tMemory used ", memory_usage, "/", memory_capacity)
    print("\tExplo ", epsilon)

def print_new_model(new_epoch):
    if new_epoch:
        print("Reloading model .", end='', flush=True)
    else :
        print(".", end='', flush=True) # count the dots !
    return False

def get_frames_from_game(env, dqn):
    observation = env.reset()
    done = False
    frames = []
    while not done:
        frames.append(env.render(mode = 'rgb_array'))
        action = eps_greedy(0.05,env.action_space.n, dqn, observation)
        # env step
        observation, reward, done, info = env.step(action)
        observation = observation.astype(float)/255

    return frames

def display_gif_from_frames(frames):
    """ Deprecated function; used to send a gif for HTML """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    animate = lambda i: patch.set_data(frames[i])
    ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval = 50)
    return ani


def keep_playing(game, test_explo, dqn, agent_history_length, new_algo):
    print("Now showing off them mad skillz")
    env = gym.make(game) # environment
    env = wrap_dqn(env)
    n_actions = env.action_space.n
    replay_memory = []
    max_memory = agent_history_length
    score_record = [] # episode scores over time (episodes)
    frame = 0
    ## Play forever
    while True:
        # init observation
        observation = env.reset().squeeze(axis=2)
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
            cumul_score += reward
            frame += 1

            # replay memory handling
            replay_memory.append((observation.squeeze(axis=2), action, reward, done))
            if len(replay_memory) > max_memory:
                del replay_memory[0]

        print("\tScore on this episode : ", cumul_score)
