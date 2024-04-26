# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:05:37 2024

@author: david
"""

import numpy as np
import math

from itertools import product
import time

def get_relative_indices(state_action, ns, iv, dim_state, dim_action):
  if state_action == 'state':
    iv_idx = dim_state + iv
  if state_action == 'action':
    iv_idx = 2 * dim_state + iv

  k_idx = [x for x in np.arange(dim_state, 2*dim_state+dim_action) if x != iv_idx]
  #couple_indices = np.array([ns, iv_idx])

  return ns, iv_idx, k_idx

def compute_MI_entry(iv_label, ns_idx, iv_idx, n, m, history):
  #delta = 1e-2  
  #hoeffding = math.sqrt(math.log(1/delta)/len(history))
  
  ns, iv, k_idx = get_relative_indices(iv_label, ns_idx, iv_idx, n, m)

  couple_idx = np.array([ns, iv])
  indices_3v = np.concatenate((couple_idx, k_idx))
  indices_ns = np.concatenate((np.array([ns]), k_idx))
  indices_iv = np.concatenate((np.array([iv]), k_idx))

  unique_3v, counts_3v = np.unique(np.asarray(history)[:, indices_3v], return_counts=True, axis=0)
  unique_ns, counts_ns = np.unique(np.asarray(history)[:, indices_ns], return_counts=True, axis=0)
  unique_iv, counts_iv = np.unique(np.asarray(history)[:, indices_iv], return_counts=True, axis=0)
  unique_r, counts_r = np.unique(np.asarray(history)[:, k_idx], return_counts=True, axis=0)
  
  unique_ns_tuples = [tuple(x) for x in unique_ns]
  unique_iv_tuples = [tuple(x) for x in unique_iv]
  unique_r_tuples = [tuple(x) for x in unique_r]
  
  dict_ns = {key: value for key, value in zip(unique_ns_tuples, counts_ns)}
  dict_iv = {key: value for key, value in zip(unique_iv_tuples, counts_iv)}
  dict_r = {key: value for key, value in zip(unique_r_tuples, counts_r)}

  mi = 0
  for arr, count in zip(unique_3v, counts_3v): 
    ns_value = arr[0]
    iv_r_value = arr[1:]
    r_value = arr[2:]
    ns_r_value = np.concatenate((np.array([ns_value]), r_value))

    c_ns = dict_ns.get(tuple(ns_r_value), 0)
    c_iv = dict_iv.get(tuple(iv_r_value), 0)
    c_r = dict_r.get(tuple(r_value), 0)

    freq_3v = count/len(history)
    freq_ns = c_ns/len(history)
    freq_iv = c_iv/len(history)
    freq_r = c_r/len(history)

    mi += freq_3v * np.log((freq_3v * freq_r)/(freq_ns * freq_iv))

  return mi

def compute_cmi_matrix(n, m, history):
    
    MI = np.zeros((n, n+m))
    
    #lp = 1e-18
    st = time.time()
    for ns in range(n):
      print()
      print('--------------------')
      print(f'Next state {ns}/{n}')
      iv_label = 'state'
      #dom_ns, dom_iv, dom_r = create_domains(iv_label, n, m, dim_state, dim_action)
      for cs in range(n):
        sti = time.time()  
        print(f'Input variable: state {cs}/{n}')  
       
        MI[ns][cs] = compute_MI_entry(iv_label, ns, cs, n, m, history)
        print(f'Computed probabilities. Elapsed time: {round(time.time()-sti, 2)} s')
        
    
      iv_label = 'action'
      #dom_ns, dom_iv, dom_r = create_domains(iv_label, n, m, dim_state, dim_action)
      for a in range(m):
        sti = time.time() 
        print(f'Input variable: action {a}/{m}')  
        
        MI[ns][n+a] = compute_MI_entry(iv_label, ns, a, n, m, history)
        print(f'Computed probabilities. Elapsed time: {round(time.time()-sti, 2)} s')
     
    print('-----------------------------------------')    
    print(f'Total time: {round(time.time() - st, 2)} s')
    return MI