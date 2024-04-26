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
    
    env_name = "rte_case5_example"
    n_epochs = 2500
    threshold = 0.01

    history, n, m = de.run(env_name, n_epochs)
    
    cmi_matrix = cmi.compute_cmi_matrix(n, m, history)
    
    '''targets = []
    variables = []
    for i in range(n):
        targets.append('s{}\''.format(i))
        variables.append('s{}'.format(i))
        
    for i in range(m):
        variables.append('a{}'.format(i))
        
    block_matrix, blocks = bd.block_diagonalization(cmi_matrix, targets, variables, threshold)'''
    
    with open('attempt_on_grid2op.npy', 'wb') as f:
        np.save(f, cmi_matrix)