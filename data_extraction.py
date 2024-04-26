# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:00:08 2024

@author: david
"""

import grid2op
from grid2op.Agent import RandomAgent
from grid2op.Reward import CloseToOverflowReward
from simulator_1 import FactoredMDP_Env_prof
import numpy as np
import math

from itertools import product
import time

def discretize(vec, bin_size = 0.05):
    discretized_values = np.zeros(len(vec))
    
    for i in range(len(vec)):
        bin_index = int(vec[i] / bin_size)
    
        if bin_index < 0:
            bin_index = 0
        if bin_index > 1/bin_size:
            bin_index = 20
        
        discretized_values[i] = bin_index
        
    return discretized_values

def run(env_name, epochs):
    env = grid2op.make(env_name, reward_class=CloseToOverflowReward, test=True)
    obs = env.reset()

    agent = RandomAgent(env.action_space)
    env.seed(0)  # for reproducible experiments
    bin_size = 0.05
    
    ###################################
    env.chronics_handler.set_chunk_size(100)
    ###################################

    episode_count = epochs  # i want to make lots of episode

    # i initialize some useful variables
    H = 0
    reward = 0
    done = False
    total_reward = 0
    
    n = len(obs.rho)
    action = agent.act(obs, reward, done)
    m = len(action._set_topo_vect)

    history = []
    # and now the loop starts
    # it will only used the chronics selected
    for i in range(episode_count):
        _ = env.chronics_handler.sample_next_chronics()
        ob = env.reset()
        
        rho = discretize(ob.rho, bin_size)

        # now play the episode as usual
        while True:
            H += 1
            
            curr_rho = rho
            action = agent.act(ob, reward, done)
            
            ob, reward, done, info = env.step(action)
            rho = discretize(ob.rho, bin_size)
            print(reward)
            
            history_entry = np.concatenate((rho, curr_rho, action._set_topo_vect))
            history.append(history_entry)
            
            total_reward += reward
            if done:
                # in this case the episode is over
                break
    return history, n, m