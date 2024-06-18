# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:58:35 2024

@author: david
"""

import numpy as np

import data_extraction as de
import cmi_computation as cmi
#import block_diag as bd

if __name__=='__main__':
       
    env_name = "l2rpn_case14_sandbox"
    #env_name = "rte_case5_example"
    n_samples = 10
    
    mi_matrix = np.zeros((20, 14))  # states x n_substations
    extraction_times = []
    computation_times = []

    history, n, m, t = de.run(env_name, n_samples)

    mi_matrix = cmi.compute_mi_matrix_parallel(n, m, np.asarray(history))
        
    with open('mi.npy', 'wb') as f:
        np.save(f, mi_matrix)
        

    
        