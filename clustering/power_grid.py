# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:58:35 2024

@author: david
"""

import numpy as np

import extract_data.runner as runner
import extract_data.fetch_data as fd

import mutual_information.cmi_computation as cmi

import grid2op
import os

import cluster.block_diag as bd

SEED = 29

if __name__=='__main__':

    # Inputs: modify as will
    env_name = 'l2rpn_case14_sandbox'
    n_episodes = 1
    n_samples = 10

    quant_list = [
    #   .50, .55, .56, .57, .58, .59,
    #   .60, .61, .62, .63, .64, .65, .66, .67, .68, .69, 
    #   .70, .71, .72, .73, .74, .75, .76, .77, .78, .79,
    #   .80, .81, .82, .83, .84, .85, .86, .86, .88, .89,
      .90, .91, .92, .93, .94,
    ]

    # --------------------------------------------
    
    path = f"./data/{env_name}_{n_episodes}"
    env = grid2op.make(env_name)

    n = env.observation_space.n_line
    m = env.observation_space.n_sub

    connections = env.action_space.sub_info
    #collect history for each substation
    mi = np.zeros((n,m))
    shuffled_mi = np.zeros((n,m))

    for sub in range(m):
        if connections[sub] > 3:
            sub_path = os.path.join(path, f'sub{sub}')
            os.makedirs(sub_path, exist_ok=True)

            runner.run(sub, env, n_episodes, SEED, sub_path)

            fd.fetch(env, n_samples, sub_path)

            history = np.load(os.path.join(sub_path, "hist.npz"))["data"]
            mi_vector, eta = cmi.compute_mi_matrix_parallel(n, m, sub, history)
            mi[:,sub] = mi_vector[:,n+sub]

            shuffled_history = history.copy()
            np.random.shuffle(shuffled_history[:,:n])
            shuffled_vector, seta = cmi.compute_mi_matrix_parallel(n, m, sub, shuffled_history)
            shuffled_mi[:,sub] = shuffled_vector[:,n+sub]
        
    unbiased_mi = mi - shuffled_mi

    with open(os.path.join(path, 'mi.npy'), 'wb') as f:
        np.save(f, mi)

    with open(os.path.join(path, 'shuffled_mi.npy'), 'wb') as f:
        np.save(f, shuffled_mi)

    with open(os.path.join(path, 'unbiased_mi.npy'), 'wb') as f:
        np.save(f, unbiased_mi)

    bd.diagonalize(unbiased_mi, os.path.join(path, 'diagonalizations'), quant_list, env_name)


    
        
