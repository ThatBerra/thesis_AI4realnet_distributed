# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:01:14 2024

@author: david
"""

import gymnasium
from gymnasium import spaces
import numpy as np

import mutual_information.mixed as mixed
import cluster.block_diag as bd
import multiprocessing as mp

from tqdm import tqdm
from itertools import product
import time
import os

SEED = 90566
np.random.seed(SEED)

'''
1st type of environment.
It keeps a relation matrix, at each steps for each state variable it randomly selects one of the related variables 
and copies its value.
ASSUMPTION: state variables and action variables can take values in the same domain
'''


class Synthetic_environment(gymnasium.Env):

  def __init__(self, relation_matrix, n, m, horizon, state_range, action_range):
    self.H = horizon
    self.t = 0
    self.history = []

    self.m = m # number of action variables
    self.n = n #number of state variables
    self.relations = self.find_relations(relation_matrix)
    self.state_range = state_range
    self.action_range = action_range

    self.observation_space = spaces.Box(low=state_range[0], high=state_range[1], shape=(self.n,), dtype=np.int64)
    self.action_space = spaces.Box(low=action_range[0], high=action_range[1], shape=(self.m,), dtype=np.int64)

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


'''
2nd type of environment.
It defines a transaction probability matrix and draws the new values of each state variable from it
'''
class FactoredMDP_Env(gymnasium.Env):

  def __init__(self, transition_prob, n, m, horizon, state_range, action_range):
    #Time-related parameters
    self.H = horizon
    self.t = 0
    self.history = []

    #We suppose to have a (n+m)xn matrix, where n is the number of state variables and m the number of action variables
    self.transition_prob = transition_prob
    self.m = m # number of action variables
    self.n = n #number of state variables
    self.state_range = state_range
    self.action_range = action_range

    self.observation_space = spaces.Box(low=state_range[0], high=state_range[1], shape=(self.n,), dtype=np.int64)
    self.action_space = spaces.Box(low=action_range[0], high=action_range[1], shape=(self.m,), dtype=np.int64)

    #we initialize the state as a random n-uple calling the reset function
    self.current_state = self.reset()

  def _get_obs(self):
    return self.current_state

  def _get_history(self):
    return self.history

  def reset(self, seed=None):
    super().reset(seed=seed)
    return np.random.randint(self.state_range[0], self.state_range[1] + 1, size=self.n)

  def get_next_state_prob(self, a):
        
    ptrans = self.transition_prob[0]
    domain_sa = self.transition_prob[1][0]
    
    sa = np.concatenate((self.current_state, a))
    ridx = domain_sa.index(tuple(sa)) # row index of (s,a)
    
    return ptrans[ridx,:]

  def get_next_state(self, a):
        
    ptrans = self.get_next_state_prob(a)
    domain_sprime = self.transition_prob[1][1]
    sprime_idx = np.random.choice(np.arange(len(domain_sprime)), p=ptrans)
    
    return np.asarray(domain_sprime[sprime_idx])

  def step(self, action):
    self.t += 1

    next_state = self.get_next_state(action)

    h_entry = np.concatenate((next_state, self.current_state, action))

    self.history.append(h_entry)
    self.current_state = next_state

    reward = "..."
    terminated = False
    if(self.t==self.H):
      terminated=True

    return self._get_obs, reward, terminated


def gen_distr(l):
    p = np.zeros(l)
    for i in range(l):
        high = 1 - np.sum(p)
        if i == l-1:
            p[i] = high
        else:
            p[i] = high * np.random.random()
    return p

def create_prob_matrix(nrows, ncol):
    prob = np.zeros((nrows, ncol))
    for i in range(nrows):
        prob[i,:] = gen_distr(ncol)
    return prob

def gen_blocks_transition_prob(blocks, dim_s, dim_a):
    
    transition_prob_blocks = []
    for block in blocks:
        ns = len(block[0]) # number of state variables in the block
        na = len(block[1]) # number of action variables in the block
        dom_ns_size = dim_s ** ns  # size of the domain of the s'
        dom_sa_size = dom_ns_size * (dim_a ** na)  # size of the domain of (s,a)
    
        pi = create_prob_matrix(nrows=dom_sa_size, ncol=dom_ns_size)
        dom_ns = list(product(*[np.arange(dim_s) for _ in range(ns)]))  # domain s'
        dom_sa = list(product(*[np.arange(dim_s) for _ in range(ns)] + [np.arange(dim_a) for _ in range(na)]))  # domain (s,a)
    
        transition_prob_blocks.append((pi, [dom_sa, dom_ns]))
        
    return transition_prob_blocks

def gen_global_transition_prob(blocks, dim_s, dim_a, len_s, len_a):
    
    transition_prob_blocks = gen_blocks_transition_prob(blocks, dim_s, dim_a)
    
    dom_ns_size = dim_s ** len_s
    dom_sa_size = dom_ns_size * (dim_a ** len_a)
    
    dom_ns = list(product(*[np.arange(dim_s) for _ in range(len_s)]))
    dom_sa = list(product(*[np.arange(dim_s) for _ in range(len_s)] + [np.arange(dim_a) for _ in range(len_a)]))
    
    ptrans = np.ones((dom_sa_size, dom_ns_size))
    for i in range(dom_sa_size):
        for j in range(dom_ns_size):
            sa = np.asarray(dom_sa[i])
            sprime = np.asarray(dom_ns[j])
            for block_idx, block in enumerate(blocks):
                sablock = np.concatenate((sa[:len_s][block[0]], sa[len_s:][block[1]])) # (s,a) for this block
                sprimeblock = sprime[block[0]] # s' for this block
                
                prob = transition_prob_blocks[block_idx][0]
                domains = transition_prob_blocks[block_idx][1]
    
                ridx = domains[0].index(tuple(sablock))  # find row index of (s,a)
                cidx = domains[1].index(tuple(sprimeblock))  # find column index of s'
    
                ptrans[i][j] *= prob[ridx][cidx]
                
    return (ptrans, [dom_sa, dom_ns])

def get_relative_indices(state_action, ns, iv, dim_state, dim_action):
  if state_action == 'state':
    iv_idx = dim_state + iv
  if state_action == 'action':
    iv_idx = 2 * dim_state + iv

  k_idx = [x for x in np.arange(dim_state, 2*dim_state+dim_action) if x != iv_idx]

  return ns, iv_idx, k_idx

def compute_MI_entry(iv_label, ns_idx, iv_idx, n, m, history):
  ns, iv, k_idx = get_relative_indices(iv_label, ns_idx, iv_idx, n, m)
  
  ns_vector = history[:, ns].reshape((len(history),1))
  iv_vector = history[:, iv].reshape((len(history),1))

  print(f"[{os.getpid()}] : starting Mixed_KSG", flush=True)
  st_time = time.time()
  mi_ns_iv = mixed.Mixed_KSG(ns_vector, iv_vector, k=int(len(history)/20))
  end_time = time.time()
  print(f'[{os.getpid()}] : ETA {round(end_time-st_time,2)}. Next state {ns}/{n}. Input variable: {iv_idx}', flush=True)  

  return mi_ns_iv

def compute_MI_entry_wrapper(args):
    return compute_MI_entry(*args)

def compute_mi_matrix_parallel(n, m, history):
    MI = np.zeros((n, n+m))

    history = np.asanyarray(history)

    st = time.time()

    pool = mp.Pool(55)

    args_list = []
    for ns in range(n):
        iv_label = 'state'
        for cs in range(n):
            args_list.append((iv_label, ns, cs, n, m, history))

        iv_label = 'action'
        for a in range(m):
                args_list.append((iv_label, ns, a, n, m, history))

    results = []
    for result in tqdm(pool.imap(compute_MI_entry_wrapper, args_list), total=len(args_list)):
        results.append(result)

    i = 0
    for ns in range(n):
        for cs in range(n):
             MI[ns][cs] = results[i]
             i += 1

        for a in range(m):
                MI[ns][n+a] = results[i]
                i += 1

    print('-----------------------------------------')
    print(f'Total time: {round(time.time() - st, 2)} s')

    t = round(time.time() - st, 2)
    return MI, t   

if __name__=='__main__':
    
    n_epochs = 2500
    
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
    
    relation_matrix = np.array([
        [1, 0, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 0]
        ])
    
    H = 100000

    done = False
    env = Synthetic_environment(relation_matrix, n, m, H, state_range, action_range)
    
    while not done:
      action = np.random.random(size=m)
      next_observation, reward, done = env.step(action)
      
    print('EXTRACTED DATA')
    
    history = env._get_history()  # H triples (s, a, s')
    
    cmi_matrix, t = compute_mi_matrix_parallel(n, m, np.asarray(history))
    
    path = 'data/custom_env'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'mi.npy'), 'wb') as f:
        np.save(f, cmi_matrix)

    thres_list = [
      #   .50, .55, .56, .57, .58, .59,
      #   .60, .61, .62, .63, .64, .65, .66, .67, .68, .69, 
       .70, .71, .72, .73, .74, .75, .76, .77, .78, .79,
       .80, .81, .82, .83, .84, .85, .86, .86, .88, .89,
      .90, .91, .92, .93, .94,
      ]

    bd.diagonalize_synthetic(cmi_matrix, 'data/custom_env/diagonalizations', thres_list)

    