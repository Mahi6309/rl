# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:42:59 2023

@author: HP
"""


#TD PREDICTION
import gymnasium as gym
import pandas as pd

#create the envt 
env = gym.make("FrozenLake-v1", render_mode = "human")
env.reset()

env.render() 

def random_policy():
    return env.action_space.sample()

V = {}
for s in range(env.observation_space.n):
    V[s] = 0.0 


alpha = 0.85
gamma = 0.90

num_eps = 50
num_steps = 10

for i in range(num_eps): 
    s = env.reset()
    s = s[0]
    for t in range(num_steps): 
        a = random_policy()
        s_, r, done, _, _ = env.step(a) 
        
                    
        V[s] += alpha * (r + gamma * V[s_] - V[s])             
        s = s_
        if done: 
           break

df = pd.DataFrame(list(V.items()), columns = ['state', 'value'])
print(df)
