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

#TODO change the name of this class
class FactoredMDP_prof_Env(gymnasium.Env):

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
  #couple_indices = np.array([ns, iv_idx])

  return ns, iv_idx, k_idx

def compute_MI_entry(iv_label, ns_idx, iv_idx, n, m, history):
  #TODO Clean this function
  '''delta = 0.1  
  hoeffding = math.sqrt(math.log(1/delta)/len(history))
  
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
    lower_3v = count/len(history) - hoeffding
    
    if lower_3v > 0:
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
    
        lower_ns = count/c_ns - hoeffding
        lower_r = c_r/c_iv

        mi += lower_3v * np.log(lower_ns * lower_r)

        #mi += freq_3v * np.log((freq_3v * freq_r)/(freq_ns * freq_iv))'''
    
  
  ns, iv, k_idx = get_relative_indices(iv_label, ns_idx, iv_idx, n, m)
  
  ns_vector = history[:, ns].reshape((len(history),1))
  iv_vector = history[:, iv].reshape((len(history),1))
  #sa_vector = history[:, n:]
  #k_vector = history[:, k_idx]
  
  #mi_ns_sa = mixed.Mixed_KSG(ns_vector, sa_vector, k=int(len(history)/20))
  #mi_ns_k = mixed.Mixed_KSG(ns_vector, k_vector, k=int(len(history)/20))
  
  print(f"[{os.getpid()}] : starting Mixed_KSG", flush=True)
  st_time = time.time()
  mi_ns_iv = mixed.Mixed_KSG(ns_vector, iv_vector, k=int(len(history)/20))
  end_time = time.time()
  print(f'[{os.getpid()}] : ETA {round(end_time-st_time,2)}. Next state {ns}/{n}. Input variable: {iv_idx}', flush=True)  

  #return mi_ns_sa - mi_ns_k
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
    
    #TODO Remove comments. Make clear the distinction between input parameters and what should not be touched
    
    #with open('cmi_custom1_cont_mixture_3_10M.npy', 'rb') as f:
        #cmi_m = np.load(f)
    
    env_name = "rte_case5_example"
    n_epochs = 2500
    #threshold = 0.01
    
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
   
    #transition_prob = gen_global_transition_prob(blocks, dim_state, dim_action, n, m)
    
    H = 100000

    done = False
    env = FactoredMDP_prof_Env(relation_matrix, n, m, H, state_range, action_range)
    
    while not done:
      action = np.random.random(size=m)
      #action = np.random.randint(action_range[0], action_range[1], size=m)
      next_observation, reward, done = env.step(action)
      
    print('EXTRACTED DATA')
    
    history = env._get_history()  # H triples (s, a, s')
    
    #cmi_matrix = mix.compute_cmi_mixture(history, n, m, k)
    cmi_matrix, t = compute_mi_matrix_parallel(n, m, np.asarray(history))
    
    '''targets = []
    variables = []
    for i in range(n):
        targets.append('s{}\''.format(i))
        variables.append('s{}'.format(i))
        
    for i in range(m):
        variables.append('a{}'.format(i))
        
    block_matrix, blocks = bd.block_diagonalization(cmi_matrix, targets, variables, threshold)'''
    path = 'data/custom_env'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'mi.npy'), 'wb') as f:
        np.save(f, cmi_matrix)

    bd.diagonalize_synthetic(cmi_matrix, 'data/custom_env/diagonalizations')

    