# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:01:14 2024

@author: david
"""

import gym
from gym import spaces
import numpy as np

from itertools import product
import time


class FactoredMDP_Env_prof(gym.Env):

  def __init__(self, relation_matrix, n, m, horizon, state_range, action_range):
    #Time-related parameters
    self.H = horizon
    self.t = 0
    self.history = []

    #We suppose to have a (n+m)xn matrix, where n is the number of state variables and m the number of action variables
    self.m = m # number of action variables
    self.n = n #number of state variables
    self.relations = self.find_relations(relation_matrix)
    self.state_range = state_range
    self.action_range = action_range

    self.observation_space = spaces.Box(low=state_range[0], high=state_range[1], shape=(self.n,), dtype=np.int64)
    self.action_space = spaces.Box(low=action_range[0], high=action_range[1], shape=(self.m,), dtype=np.int64)

    #we initialize the state as a random n-uple calling the reset function
    self.current_state = self.reset()

  def find_relations(self, relation_matrix):
    relations = []

    for i in range(self.n):
      idxs = [_ for _, x in enumerate(relation_matrix[i]) if x==1]
      relations.append(idxs)

    return relations

  def _get_obs(self):
    return self.current_state

  def _get_history(self):
    return self.history

  def reset(self, seed=None):
    super().reset(seed=seed)
    return np.random.randint(self.state_range[0], self.state_range[1] + 1, size=self.n)

  def get_next_state(self, current_state, action):
    next_state = np.zeros(self.n)
    input_variables = np.concatenate((current_state, action))
    for i in range(self.n):
      idxs = self.relations[i]
      related_variables = np.asarray(input_variables)[idxs]
      next_state[i] = np.random.choice(related_variables)

    return next_state

  def step(self, action):
    self.t += 1

    next_state = self.get_next_state(self.current_state, action)

    #h_entry = [self.current_state, action, next_state]
    h_entry = np.concatenate((next_state, self.current_state, action))

    self.history.append(h_entry)
    self.current_state = next_state

    reward = "..."
    terminated = False
    if(self.t==self.H):
      terminated=True

    return self._get_obs, reward, terminated


def create_domains(state_action, n, m, dim_state, dim_action):
  dom_ns = np.arange(dim_state)

  if state_action == 'state':
    dom_iv = np.arange(dim_state)
    dom_r = list(product(*[np.arange(dim_state) for _ in range(n-1)] + [np.arange(dim_action) for _ in range(m)]))

  if state_action == 'action':
    dom_iv = np.arange(dim_action)
    dom_r = list(product(*[np.arange(dim_state) for _ in range(n)] + [np.arange(dim_action) for _ in range(m-1)]))

  return dom_ns, dom_iv, dom_r


def get_relative_indices(state_action, ns, iv, dim_state, dim_action):
  if state_action == 'state':
    iv_idx = dim_state + iv
  if state_action == 'action':
    iv_idx = 2 * dim_state + iv

  k_idx = [x for x in np.arange(dim_state, 2*dim_state+dim_action) if x != iv_idx]
  #couple_indices = np.array([ns, iv_idx])

  return ns, iv_idx, k_idx


def compute_3var_frequencies(state_action, ns_idx, iv_idx, dom_ns, dom_iv, dom_r, dim_state, dim_action, history):
  frequency_matrix = np.zeros((len(dom_ns), len(dom_iv), len(dom_r)))

  ns, iv, k_idx = get_relative_indices(state_action, ns_idx, iv_idx, dim_state, dim_action)
  couple_idx = np.array([ns, iv])
  indices = np.concatenate((couple_idx, k_idx))

  unique_transactions, counts = np.unique(np.asarray(history)[:, indices], return_counts=True, axis=0)
  for arr, count in zip(unique_transactions, counts):
    ns_value = arr[0]
    iv_value = arr[1]
    r = arr[2:]

    i = np.where(dom_ns == ns_value)[0][0]
    j = np.where(dom_iv == iv_value)[0][0]
    k =[_ for _, x in enumerate(dom_r) if np.array_equal(np.asarray(dom_r[_]), r)][0]

    frequency_matrix[i][j][k] = count/len(history)
  
  return frequency_matrix


def compute_2var_ns_frequencies(state_action, ns_idx, iv_idx, dom_ns, dom_r, dim_state, dim_action, history):
  frequency_matrix = np.zeros((len(dom_ns), len(dom_r)))

  ns, iv, k_idx = get_relative_indices(state_action, ns_idx, iv_idx, dim_state, dim_action)
  indices = np.concatenate((np.array([ns]), k_idx))

  unique_transactions, counts = np.unique(np.asarray(history)[:, indices], return_counts=True, axis=0)

  for arr, count in zip(unique_transactions, counts):
    ns_value = arr[0]
    r = arr[1:]

    i = np.where(dom_ns == ns_value)[0][0]
    k =[_ for _, x in enumerate(dom_r) if np.array_equal(np.asarray(dom_r[_]), r)][0]

    frequency_matrix[i][k] = count/len(history)
  
  return frequency_matrix


def compute_2var_cv_frequencies(state_action, ns_idx, iv_idx, dom_iv, dom_r, dim_state, dim_action, history):
  frequency_matrix = np.zeros((len(dom_iv), len(dom_r)))

  ns, iv, k_idx = get_relative_indices(state_action, ns_idx, iv_idx, dim_state, dim_action)
  indices = np.concatenate((np.array([iv]), k_idx))

  unique_transactions, counts = np.unique(np.asarray(history)[:, indices], return_counts=True, axis=0)

  for arr, count in zip(unique_transactions, counts):
    iv_value = arr[0]
    r = arr[1:]

    j = np.where(dom_iv == iv_value)[0][0]
    k =[_ for _, x in enumerate(dom_r) if np.array_equal(np.asarray(dom_r[_]), r)][0]

    frequency_matrix[j][k] = count/len(history)
  
  return frequency_matrix


def compute_remainder_marginal(state_action, ns_idx, iv_idx, dom_r, dim_state, dim_action, history):
  frequency_array = np.zeros(len(dom_r))

  ns, iv, k_idx = get_relative_indices(state_action, ns_idx, iv_idx, dim_state, dim_action)

  unique_transactions, counts = np.unique(np.asarray(history)[:, k_idx], return_counts=True, axis=0)

  for arr, count in zip(unique_transactions, counts):
    k =[_ for _, x in enumerate(dom_r) if np.array_equal(np.asarray(dom_r[_]), arr)][0]

    frequency_array[k] = count/len(history)
  
  return frequency_array


def compute_cmi_matrix(n, m, dim_state, dim_action, history):
    
    MI = np.zeros((n, n+m))
    
    lp = 1e-18
    st = time.time()
    for ns in range(n):
      print()
      print('--------------------')
      print(f'Next state {ns}/{n}')
      iv_label = 'state'
      dom_ns, dom_iv, dom_r = create_domains(iv_label, n, m, dim_state, dim_action)
      for cs in range(n):
        sti = time.time()  
        print(f'Input variable: state {cs}/{n}')  
        s_3var_freq = compute_3var_frequencies(iv_label, ns, cs, dom_ns, dom_iv, dom_r, n, m, history)
        ns_2v_freq = compute_2var_ns_frequencies(iv_label, ns, cs, dom_ns, dom_r, n, m, history)
        cs_2v_freq = compute_2var_cv_frequencies(iv_label, ns, cs, dom_iv, dom_r, n, m, history)
        remainder_marginal = compute_remainder_marginal(iv_label, ns, cs, dom_r, n, m, history)
        print(f'Computed probabilities. Elapsed time: {round(time.time()-sti, 2)} s')
        
        s = 0
        for i in range(len(dom_ns)):
          for j in range(len(dom_iv)):
            for k in range(len(dom_r)):
              #l = max(s_3var_freq[i][j][k] * remainder_marginal[k], lp)/(max(cs_2v_freq[j][k] * ns_2v_freq[i][k], lp))  
              #w = s_3var_freq[i][j][k] * np.log(max(s_3var_freq[i][j][k] * remainder_marginal[k], lp)/(max(cs_2v_freq[j][k] * ns_2v_freq[i][k], lp)))
              s += s_3var_freq[i][j][k] * np.log(max(s_3var_freq[i][j][k] * remainder_marginal[k], lp)/(max(cs_2v_freq[j][k] * ns_2v_freq[i][k], lp)))

        MI[ns][cs] = s
        
    
      iv_label = 'action'
      dom_ns, dom_iv, dom_r = create_domains(iv_label, n, m, dim_state, dim_action)
      for a in range(m):
        sti = time.time() 
        print(f'Input variable: action {a}/{m}')  
        a_3var_freq = compute_3var_frequencies(iv_label, ns, a, dom_ns, dom_iv, dom_r, n, m, history)
        ns_2v_freq = compute_2var_ns_frequencies(iv_label, ns, a, dom_ns, dom_r, n, m, history)
        a_2v_freq = compute_2var_cv_frequencies(iv_label, ns, a, dom_iv, dom_r, n, m, history)
        remainder_marginal = compute_remainder_marginal(iv_label, ns, a, dom_r, n, m, history)
        print(f'Computed probabilities. Elapsed time: {round(time.time()-sti, 2)} s')
        
        s = 0
        for i in range(len(dom_ns)):
          for j in range(len(dom_iv)):
            for k in range(len(dom_r)):
              s += a_3var_freq[i][j][k] * np.log(max((a_3var_freq[i][j][k] * remainder_marginal[k]),lp)/(max(a_2v_freq[j][k] * ns_2v_freq[i][k], lp)))

        MI[ns][n+a] = s
     
    print('-----------------------------------------')    
    print(f'Total time: {round(time.time() - st, 2)} s')
    return MI

if __name__=='__main__':
    
    n = 5  # number of state components
    m = 3  # number of action components
    
    state_range = [0,1]
    action_range = [0,1]
    
    dim_state = state_range[1] - state_range[0] + 1
    dim_action = action_range[1] - action_range[0] + 1
    
    blocks = [
    ([0,2,4], [0,1]),
    ([1,3], [2])
    ]
    
    input_matrix = np.asarray([
    [1, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 0]
    ])
    
    cmis = []
    experiments = [1000, 10000, 100000, 1000000, 10000000]

    for H in experiments:
        print(f'Experiment: {H} samples')  
        done = False
        env = FactoredMDP_Env_prof(input_matrix, n, m, H, state_range, action_range)
    
        while not done:
            action = np.random.randint(action_range[0], action_range[1]+1, size=m)
            next_observation, reward, done = env.step(action)
    
        history = env._get_history()  # H triples (s, a, s')
    
        cmi_matrix = compute_cmi_matrix(n, m, dim_state, dim_action, history)
        
        cmis.append(cmi_matrix)
    
    cmis = np.asarray(cmis)
    
    with open('cmi_matrices.npy', 'wb') as f:
        np.save(f, cmis)