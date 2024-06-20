# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:58:35 2024

@author: david
"""

import numpy as np

import data_extraction as de
import cmi_computation as cmi
import grid2op
#import block_diag as bd

if __name__=='__main__':
       
    path = "./wcci2022_30k"
    # env = grid2op.make("l2rpn_case14_sandbox")
    env = grid2op.make("l2rpn_wcci_2022")

    n = env.observation_space.n_line
    m = env.observation_space.n_sub

    print(f"n={n} (states), m={m} (actions)")

    # history, n, m, t = de.run(env_name, n_samples)
    history = np.load(f"{path}_hist.npz")["data"][:50000, :]

    mi_matrix, eta = cmi.compute_mi_matrix_parallel(n, m, history)
        
    with open('mi.npy', 'wb') as f:
        np.save(f, mi_matrix)

    with open('mi_time.txt', 'w') as f:
        f.write(str(eta))
        

    
        