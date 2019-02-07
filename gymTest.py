import gym
import time

env = gym.make('Breakout-v0')
frame = 0
for i_episode in range(20):
    observation = env.reset()
    done = False
    while not done:
        frame += 1
        env.render()
        time.sleep(.05)
        action = 1 if frame == 1 else 2
        print(action)
        observation, reward, done, info = env.step(action)
