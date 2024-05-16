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
    
    #with open('grid2op_5000_hoeffding.npy', 'rb') as f:
        #cmi_m = np.load(f)
    
    env_name = "l2rpn_case14_sandbox"
    #env_name = "rte_case5_example"
    n_epochs = 10
    #threshold = 0.01

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
    
    with open('grid2op_case14_gen4_10.npy', 'wb') as f:
        np.save(f, cmi_matrix)
        