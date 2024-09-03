from abc import abstractmethod

import numpy as np
import torch
from grid2op.Agent import BaseAgent
from grid2op.Action import BaseAction

EPSILON = 1e-6

class L2rpnAgent(BaseAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env.action_space)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_space = env.observation_space
        self.action_space = env.action_space
        self.mask = 3
        self.danger = 0.9
        self.thermal_limit = env._thermal_limit_a
        self.node_num = env.dim_topo


        #how many times the agent updates
        self.update_step = 0
        #for saving transitions
        self.agent_step = 0
    
        #PPO start update
        self.update_start = 16
        
        self.gamma = kwargs.get("gamma", 0.99)


    def is_safe(self, obs):
        for ratio in obs.rho:
            if ratio >= self.danger:
                return False
        return True

    def load_mean_std(self, mean, std):
        self.state_mean = mean
        self.state_std = std.masked_fill(std < 1e-5, 1.0)
        self.state_mean[0, sum(self.obs_space.shape[:20]) :] = 0
        self.state_std[0, sum(self.action_space.shape[:20]) :] = 1

    def state_normalize(self, s):
        s = (s - self.state_mean) / self.state_std
        return s

    def reset(self):
        self.topo = None
        self.goal = None
        self.adj = None
        self.stacked_obs = []
        self.forecast = []
        self.start_state = None
        self.start_adj = None
        self.save = False

    def hash_goal(self, goal):
        hashed = ""
        for i in goal.view(-1):
            hashed += str(int(i.item()))
        return hashed

    def get_current_state(self):
        return torch.Tensor(
            self.stacked_obs,
        )

    def act(self, obs, reward=None, done=None):
        # if reward is None we are TRAINING therefore take sample!
        is_safe = self.is_safe(obs)
        self.save = False

        # reconnect powerline when the powerline is disconnected
        if False in obs.line_status:
            act = self.reconnect_line(obs)
            if act is not None:
                if self.training:
                    return None, act, None, None, None
                else:
                    return act

        return self.agent_act(obs, is_safe)

    def reconnect_line(self, obs):
        # if the agent can reconnect powerline not included in controllable substation, return action
        # otherwise, return None
        dislines = np.where(obs.line_status == False)[0]
        for i in dislines:
            act = None
            if obs.time_next_maintenance[i] != 0:  # REMOVED: check for lonely lines
                sub_or = self.action_space.line_or_to_subid[i]
                sub_ex = self.action_space.line_ex_to_subid[i]
                if obs.time_before_cooldown_sub[sub_or] == 0:
                    act = self.action_space({"set_bus": {"lines_or_id": [(i, 1)]}})
                if obs.time_before_cooldown_sub[sub_ex] == 0:
                    act = self.action_space({"set_bus": {"lines_ex_id": [(i, 1)]}})
                if obs.time_before_cooldown_line[i] == 0:
                    status = self.action_space.get_change_line_status_vect()
                    status[i] = True
                    act = self.action_space({"change_line_status": status})
                if act is not None:
                    return act
        return None

    @abstractmethod
    def agent_act(self, obs, is_safe) -> BaseAction:
        pass

    def check_start_update(self):
        return len(self.memory) >= self.update_start

    def unpack_batch(self, batch):
        states, adj, actions, rewards, states2, adj2, dones, steps = list(zip(*batch))

        states = torch.cat(states, 0)
        states2 = torch.cat(states2, 0)
        adj = torch.stack(adj, 0)
        adj2 = torch.stack(adj2, 0)
        actions = torch.stack(actions, 0)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        steps = torch.FloatTensor(steps).unsqueeze(1)
        return (
            states.to(self.device),
            adj.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            states2.to(self.device),
            adj2.to(self.device),
            dones.to(self.device),
            steps.to(self.device),
        )

