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
    n_samples = 60000
    
    mi_matrix = np.zeros((20, 14))
    extraction_times = []
    computation_times = []
    for i in range(14):
        history, n, m, t = de.run(env_name, n_samples, i)
        extraction_times.append(t)
        
        if len(history) > 0:
            mi, t = cmi.compute_mi_matrix_parallel(n, m, history)
            mi = mi[:,0]
        else:
            mi = np.zeros(20)
            t = 0
        
        mi_matrix[:,i] = mi
        computation_times.append(t)
        
    with open('mi.npy', 'wb') as f:
        np.save(f, mi_matrix)
        
    with open('extract_times.npy', 'wb') as f:
        np.save(f, extraction_times)
        
    with open('compute_times.npy', 'wb') as f:
        np.save(f, extraction_times)

    
        