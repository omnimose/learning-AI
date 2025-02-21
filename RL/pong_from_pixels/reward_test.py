""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
# import gym
import gymnasium as gym

gamma = 0.99 # discount factor for reward

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


drs = [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
epr = np.vstack(drs)
print(epr)
print(epr.shape)

discountRewards = discount_rewards(epr)
print(discountRewards)
print(discountRewards.shape)


drs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
epr = np.vstack(drs)
print(epr)
print(epr.shape)

discountRewards = discount_rewards(epr)
print(discountRewards)
print(discountRewards.shape)
