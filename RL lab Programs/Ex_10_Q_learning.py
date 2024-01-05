# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:15:02 2023

@author: HP
"""


import gymnasium as gym
import numpy as np
import random
import pandas as pd

env = gym.make("FrozenLake-v1", render_mode = "human")

Q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s,a)] = 0.0

def epsilon_greedy(state, epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x: Q[(state,x)])

alpha = 0.85
gamma = 0.90
epsilon = 0.8

num_eps = 50
num_steps = 100

for i in range(num_eps):
    s = env.reset()
    s = s[0] 
    for t in range(num_steps):
        a = epsilon_greedy(s,epsilon)
        
        s_,r,done, _, _ = env.step(a)
        a_ = np.argmax([Q[(s,a)]  for a in range(env.action_space.n)])
        
        Q[(s,a)] += alpha * ( r + gamma * Q[(s_,a_)] - Q[(s,a)])
        
        s = s_
        if done:
            break

df = pd.DataFrame(list(Q.items()), columns = ['state=action', 'value'])   
print(df)                      


                   
               