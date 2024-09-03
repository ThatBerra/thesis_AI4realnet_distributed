# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:58:35 2024

@author: david
"""

import numpy as np

import clustering.extract_data.runner as runner
import clustering.extract_data.fetch_data as fd

import clustering.mutual_information.cmi_computation as cmi
import shuffled_mi as smi

import grid2op
import os

import clustering.block_diag as bd

if __name__=='__main__':
       
    env_name = 'l2rpn_case14_sandbox'
    n_episodes = 1000
    n_samples = 10000
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

            runner.run(sub, env, sub_path)

            fd.fetch(env, n_samples, sub_path)

            history = np.load(os.path.join(path, "hist.npz"))["data"]
            mi_vector, eta = cmi.compute_mi_matrix_parallel(n, m, sub, history)
            mi[:,sub] = mi_vector[:,n+sub]

            shuffled_history = history.copy()
            np.random.shuffle(shuffled_history[:,:20])
            shuffled_vector, seta = cmi.compute_mi_matrix_parallel(n, m, sub, shuffled_history)
            shuffled_mi[:,sub] = shuffled_vector[:,20+sub]
        
    unbiased_mi = mi - shuffled_mi

    with open(os.path.join(path, 'mi.npy'), 'wb') as f:
        np.save(f, mi)

    with open(os.path.join(path, 'shuffled_mi.npy'), 'wb') as f:
        np.save(f, shuffled_mi)

    with open(os.path.join(path, 'unbiased_mi.npy'), 'wb') as f:
        np.save(f, unbiased_mi)

    bd.diagonalize(unbiased_mi, os.path.join(path, 'diagonalizations'))


    
        