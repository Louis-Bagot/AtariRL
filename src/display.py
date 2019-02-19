import matplotlib.pyplot as plt
from dqn_functions import *

def graph(l, title):
    plt.plot(l)
    plt.ylabel(title)
    plt.show()

def print_info(frame, episode, epoch_nb, memory_usage, memory_capacity):
    print("Epoch ", epoch_nb)
    print("\tEpisode ", episode)
    print("\tFrame ", frame)
    print("\tMemory used ", memory_usage, "/", memory_capacity)

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
