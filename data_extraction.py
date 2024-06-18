# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:00:08 2024

@author: david
"""

import grid2op
from grid2op.Agent import RandomAgent
from grid2op.Reward import CloseToOverflowReward
import numpy as np
import math

from itertools import product
import time

SEED = 2404
np.random.seed(SEED)

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

def run(env_name, n_samples):
    env = grid2op.make(env_name, reward_class=CloseToOverflowReward, test=True)
    obs = env.reset()

    env.seed(SEED)  # for reproducible experiments

    
    ###################################
    env.chronics_handler.set_chunk_size(100)
    ###################################


    # i initialize some useful variables
    H = 0
    reward = 0
    done = False
    total_reward = 0
    
    n = len(obs.rho)
    m = obs.n_sub

    conf_subs = []
    for i in range(m): 
        if env.action_space.sub_info[i] > 3:
            conf_subs.append(i)
    
    history = []
    
    st = time.time()
    # and now the loop starts
    # it will only used the chronics selected
    #for i in range(episode_count):
    while H < n_samples:
        ob = env.reset()
        
        rho = ob.rho

        # now play the episode as usual
        while True:
            H += 1
            
            curr_rho = rho
            #action = agent.act(ob, reward, done)
            #action = env.action_space()
            #obj = np.random.randint(0, obs.dim_topo)
            #bus = np.random.randint(0, 2)
            #redisp = -5 + 10*np.random.rand()
            #action.redispatch = [(0, redisp)]
            sub_id = np.random.randint(0, len(conf_subs))
            sub = conf_subs[sub_id]

            act_list = env.action_space.get_all_unitary_topologies_set(env.action_space, sub_id=sub)

            act = np.random.randint(0, len(act_list))
            action = act_list[act]
            
            ob, reward, done, info = env.step(action)
            rho = ob.rho

            action_vector = np.zeros(m)
            action_vector[sub] = act+1

            ns_cs = np.concatenate((rho, curr_rho))
            history_entry = np.append(ns_cs, action_vector)
            history.append(history_entry)
            #history.append(ns_cs)
            
            total_reward += reward
            if done or H>=n_samples:
                break
    print('-----------------------------------------')    
    print('EXTRACTED DATA')
    print(f'Total time: {round(time.time() - st, 2)} s')
    t = round(time.time()-st, 2)
    
    return history, n, m, t

