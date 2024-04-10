import gym
from gym import spaces
import numpy as np

from itertools import product
import time



class FactoredMDP_Env(gym.Env):

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


def get_history_sample(history, time):
  current_state = history[time][0]
  action = history[time][1]
  next_state = history[time][2]

  return current_state, action, next_state

def get_remainder(state_action, state, action, index):
  if state_action == 'state':
    iv = state
    r = np.concatenate(([x for _, x in enumerate(state) if _ != index], action))

  if state_action == 'action':
    iv = action
    r = np.concatenate((state, [x for _, x in enumerate(action) if _ != index]))

  return iv, r

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

'''def compute_3var_frequencies(state_action, ns_idx, iv_idx, dom_ns, dom_iv, dom_r, history):
  frequency_matrix = np.zeros((len(dom_ns), len(dom_iv), len(dom_r)))

  for i in range(len(dom_ns)):
    for j in range(len(dom_iv)):
      for k in range(len(dom_r)):

        for t in range(len(history)):
          curr_state, action, next_state = get_history_sample(history, t)
          iv, r = get_remainder(state_action, curr_state, action, iv_idx)
          if next_state[ns_idx] == dom_ns[i] and iv[iv_idx] == dom_iv[j] and np.array_equal(r, np.asarray(dom_r[k])):
            frequency_matrix[i][j][k] += 1/len(history)

  return frequency_matrix
def compute_3var_frequencies(state_action, ns_idx, iv_idx, dom_ns, dom_iv, dom_r, history):
  frequency_matrix = np.zeros((len(dom_ns), len(dom_iv), len(dom_r)))

  for t in range(len(history)):
    curr_state, action, next_state = get_history_sample(history, t)
    iv, r = get_remainder(state_action, curr_state, action, iv_idx)

    i = np.where(dom_ns == next_state[ns_idx])[0][0]
    j = np.where(dom_iv == iv[iv_idx])[0][0]
    k = [_ for _, x in enumerate(dom_r) if np.array_equal(np.asarray(dom_r[_]), r)][0]

    frequency_matrix[i][j][k] += 1/len(history)

  return frequency_matrix'''
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

'''def compute_2var_ns_frequencies(state_action, ns_idx, iv_idx, dom_ns, dom_r, history):
  frequency_matrix = np.zeros((len(dom_ns), len(dom_r)))

  for i in range(len(dom_ns)):
    for k in range(len(dom_r)):

      for t in range(len(history)):
        curr_state, action, next_state = get_history_sample(history, t)

        _, r = get_remainder(state_action, curr_state, action, iv_idx)

        if next_state[ns_idx] == dom_ns[i] and np.array_equal(r, dom_r[k]):
          frequency_matrix[i][k] += 1/len(history)

  return frequency_matrix
def compute_2var_ns_frequencies(state_action, ns_idx, iv_idx, dom_ns, dom_r, history):
  frequency_matrix = np.zeros((len(dom_ns), len(dom_r)))

  for t in range(len(history)):
    curr_state, action, next_state = get_history_sample(history, t)
    iv, r = get_remainder(state_action, curr_state, action, iv_idx)

    i = np.where(dom_ns == next_state[ns_idx])[0][0]
    k = [_ for _, x in enumerate(dom_r) if np.array_equal(np.asarray(dom_r[_]), r)][0]

    frequency_matrix[i][k] += 1/len(history)

  return frequency_matrix'''
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

'''def compute_2var_cv_frequencies(state_action, iv_idx, dom_iv, dom_r, history):
  frequency_matrix = np.zeros((len(dom_iv), len(dom_r)))

  for j in range(len(dom_iv)):
    for k in range(len(dom_r)):

      for t in range(len(history)):
        curr_state, action, _ = get_history_sample(history, t)

        iv, r = get_remainder(state_action, curr_state, action, iv_idx)

        if iv[iv_idx] == dom_iv[j] and np.array_equal(r, dom_r[k]):
          frequency_matrix[j][k] += 1/len(history)

  return frequency_matrix
def compute_2var_cv_frequencies(state_action, iv_idx, dom_iv, dom_r, history):
  frequency_matrix = np.zeros((len(dom_iv), len(dom_r)))

  for t in range(len(history)):
    curr_state, action, next_state = get_history_sample(history, t)
    iv, r = get_remainder(state_action, curr_state, action, iv_idx)

    j = np.where(dom_iv == iv[iv_idx])[0][0]
    k = [_ for _, x in enumerate(dom_r) if np.array_equal(np.asarray(dom_r[_]), r)][0]

    frequency_matrix[j][k] += 1/len(history)

  return frequency_matrix'''
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


'''def compute_remainder_marginal(state_action, iv_idx, dom_r, history):
  frequency_array = np.zeros(len(dom_r))

  for k in range(len(dom_r)):

    for t in range(len(history)):
      curr_state, action, _ = get_history_sample(history, t)
      _, r = get_remainder(state_action, curr_state, action, iv_idx)

      if np.array_equal(r, dom_r[k]):
        frequency_array[k] += 1/len(history)

  return frequency_array
def compute_remainder_marginal(state_action, iv_idx, dom_r, history):
  frequency_array = np.zeros(len(dom_r))

  for t in range(len(history)):
    curr_state, action, next_state = get_history_sample(history, t)
    iv, r = get_remainder(state_action, curr_state, action, iv_idx)

    k = [_ for _, x in enumerate(dom_r) if np.array_equal(np.asarray(dom_r[_]), r)][0]

    frequency_array[k] += 1/len(history)

  return frequency_array'''
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
    
    transition_prob = gen_global_transition_prob(blocks, dim_state, dim_action, n, m)
    
    H = 100000

    done = False
    env = FactoredMDP_Env(transition_prob, n, m, H, state_range, action_range)
    
    while not done:
      action = np.random.randint(action_range[0], action_range[1]+1, size=m)
      next_observation, reward, done = env.step(action)
    
    history = env._get_history()  # H triples (s, a, s')
    
    cmi_matrix = compute_cmi_matrix(n, m, dim_state, dim_action, history)
    
    with open('cmi_large.npy', 'wb') as f:
        np.save(f, cmi_matrix)