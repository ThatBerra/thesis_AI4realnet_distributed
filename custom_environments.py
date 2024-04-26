# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:01:14 2024

@author: david
"""

import gym
from gym import spaces
import numpy as np
import math

from itertools import product
import time

SEED = 90566
np.random.seed(SEED)

'''
1st type of environment.
It keeps a relation matrix, at each steps for each state variable it randomly selects one of the related variables 
and copies its value.
ASSUMPTION: state variables and action variables can take values in the same domain
'''
class FactoredMDP_prof_Env(gym.Env):

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