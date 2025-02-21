# need to import: 
# pip install "gymnasium[atari,accept-rom-license]"
import numpy as np
import matplotlib.pyplot as plt
import pickle
# import gym
import gymnasium as gym
import ale_py



gym.register_envs(ale_py) 

env = gym.make("ALE/Pong-v5", render_mode="human")
observation, info = env.reset()

render = True
xs,hs,dlogps,drs = [],[],[],[]

while True:
  if render: env.render()

  action = np.random.choice([2,3])
  # step the environment and get new measurements
  observation, reward, done, truncated, info = env.step(action)

  if reward != 0:
    print(reward)

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    epr = np.vstack(drs)
    break

print(epr.shape)  
print(epr)

    