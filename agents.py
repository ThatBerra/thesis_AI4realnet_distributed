# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:52:59 2024

@author: david
"""

from grid2op.Agent import BaseAgent
import numpy as np

class Sub_Agent():
    def __init__(self, variables):
        self.elements_ids = variables
        
    def act(self, obs, reward, done=False):
        act_list = []
        for i in self.elements_ids:
            bus = np.random.randint(0, 2)
            element_tuple = (i, bus)
            act_list.append(element_tuple)
        
    def update():
        print('Nothing to do here for now')
    

class Bus_Forcing_Agent(BaseAgent):
    def __init__(self, action_space, blocks):
        BaseAgent.__init__(self, action_space)
        
        self.distr_agents = []
        for block in blocks:
            agent = Sub_Agent(block)
            self.distr_agents.append(agent)
            
            
    def act(self, obs, reward, done=False):
        action = []
        for agent in self.distr_agents:
            distr_action = agent.act(obs, reward, done)
            for act in distr_action:
                action.append(act)
        
        return action
        
        