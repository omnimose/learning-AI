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
observation = env.reset()

env.render()