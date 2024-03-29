# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:58:43 2023

@author: HP
"""


# LEARNING THE OPTIMAL POLICY USING SARSA IN FROZEN LAKE 

#import libraries
import gymnasium as gym
import random
import pandas as pd

#create the envt 
env = gym.make("FrozenLake-v1", render_mode = "human")
env.reset()

env.render()

#define a dictionary for the Q table. Initialize the Q value of all (s,a) pairs to 0.0

Q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s,a)] = 0.0
  

print("The initial Q table is ",Q)
#define the epsilon-greedy policy. We generate a random number from the uniform distribution of 0 to 1. 
#If teh random number is less than epsilon, we select a random action, else we select the best action.
def epsilon_greedy(state,epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x : Q[(state,x)])

# initialize alpha, gamma and epsilon 
alpha = 0.85
gamma = 0.90
epsilon = 0.8

#set the no. of episodes and the no. of steps in each episode

num_eps = 500
num_steps = 50

#compute the policy for each episode

for i in range(num_eps):
    s = env.reset() 
    s = s[0]
    a = epsilon_greedy(s,epsilon)
    for t in range(num_steps):
        s_, r, done, _, _ = env.step(a) 
        a_ = epsilon_greedy(s_, epsilon) 
        predict = Q[(s,a)]
        target = r + gamma * Q[(s_, a_)]
        
        Q[(s,a)] = Q[(s,a)] + alpha * (target - predict) 
        
        s = s_
        a = a_
        if done:
            break
df = pd.DataFrame(list(Q.items()), columns = ['state=action', 'value'])   
print(df)                   