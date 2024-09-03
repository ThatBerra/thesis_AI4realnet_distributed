import torch
import numpy as np

from .l2rpn_base_agent import L2rpnAgent
from Agents.ClusterAgents import LowLevel_PPO_agent, LowLevel_PPO_agent_complete_obs
from Agents.MiddleAgent import RuleBasedSubPicker

import utils.converters as converters

class IMARL(L2rpnAgent):
    """
    High Lecel agent that coordinates the low level agents
    """

    def __init__(self, env, sub_clusters, line_clusters, seed, **kwargs):
        super().__init__(env, **kwargs)

        subs = np.flatnonzero(self.action_space.sub_info > self.mask)
        sort_subs = np.argsort(-self.action_space.sub_info[self.action_space.sub_info > self.mask])
        masked_sorted_sub = subs[sort_subs]
        self.sub_picker = RuleBasedSubPicker(masked_sorted_sub, action_space=self.action_space)
        
        self.sub_areas = sub_clusters
        self.line_areas = line_clusters
        self.episode_rewards = []
        self.episode_survival = []
        self.training = False

        # create low level RL agents
        self.create_DLA(env, seed, **kwargs)
        self.reset()

    def create_DLA(self, env, seed, **kwargs):
        obs_attr_to_keep = ["month", "day_of_week", "hour_of_day", "minute_of_hour",
                    "gen_p",  # generators
                    "load_p",  # loads
                    "p_or", "rho", "timestep_overflow", "line_status",  # lines
                    "actual_dispatch", "target_dispatch",  # generators
                    "curtailment", "curtailment_limit",  "gen_p_before_curtail",  # generators
                    "topo_vect", "time_before_cooldown_sub"
                    ]
        agents = []
        for i, cluster in enumerate(self.sub_areas):
            converter = converters.ClusterConverter(env, obs_attr_to_keep, cluster, self.line_areas[i], seed)
            agent = LowLevel_PPO_agent(env, converter, **kwargs)
            agents.append(agent)

        self.agents = np.array(agents)

    def reset(self, obs=None):
        super().reset()
        self.sub_picker.complete_reset()

    def set_training_status(self, training):
        self.training = training

    def cache_stat(self):
        cache = super().cache_stat()
        if self.goal is not None:
            (agent, action) = self.goal
            sub_vals = self.agents[agent].cache_stat()
            cache_extra = {
                "sub_vals": sub_vals,
            }
            cache.update(cache_extra)
        return cache

    def load_cache_stat(self, cache):
        super().load_cache_stat(cache)
        if self.goal is not None:
            (sub_2_act, action) = self.goal
            sub_vals = cache["sub_vals"]
            self.agents[sub_2_act].load_cache_stat(sub_vals)

    def find_cluster(self, sub_2_act):
        for i, c in enumerate(self.sub_areas):
            if sub_2_act in c:
                return i
        return -1

    def agent_act(self, obs, is_safe):
        # generate action if not safe
        if not is_safe or (len(self.sub_picker.subs_2_act) > 0):
            with torch.no_grad():
                # Middle agent chooses a substation that needs an action at this time
                sub_2_act = self.sub_picker.pick_sub(obs)
                # Select the agent that operates on the chosen substation
                agent = self.find_cluster(sub_2_act)
                idx, action, values, log_probs = self.agents[agent].act(obs, None)
                
        else:
            action = self.action_space({})
            idx = None
            values = None
            agent = None
            log_probs = None
        
        # needed to make the agent compliant with grid2op runner 
        if self.training:
            return idx, action, agent, values, log_probs
        else:
            return action

    def save_model(self, path):
        [agent.save_model(f"{path}", f"cluster_{sub}") for sub, agent in enumerate(self.agents)]

    def load_model(self, path):
        [agent.load_model(f"{path}", f"cluster_{sub}") for sub, agent in enumerate(self.agents)]


class IMARL_complete_obs(L2rpnAgent):
    """
    High Lecel agent that coordinates the low level agents
    """

    def __init__(self, env, sub_clusters, line_clusters, seed, **kwargs):
        super().__init__(env, **kwargs)

        subs = np.flatnonzero(self.action_space.sub_info > self.mask)
        sort_subs = np.argsort(-self.action_space.sub_info[self.action_space.sub_info > self.mask])
        masked_sorted_sub = subs[sort_subs]
        self.sub_picker = RuleBasedSubPicker(masked_sorted_sub, action_space=self.action_space)
        
        self.sub_areas = sub_clusters
        self.line_areas = line_clusters
        self.episode_rewards = []
        self.episode_survival = []
        self.training = False

        # create low level RL agents
        self.create_DLA(env, seed, **kwargs)
        self.reset()

    def create_DLA(self, env, seed, **kwargs):
        obs_attr_to_keep = ["month", "day_of_week", "hour_of_day", "minute_of_hour",
                    "gen_p",  # generators
                    "load_p",  # loads
                    "p_or", "rho", "timestep_overflow", "line_status",  # lines
                    "actual_dispatch", "target_dispatch",  # generators
                    "curtailment", "curtailment_limit",  "gen_p_before_curtail",  # generators
                    "topo_vect", "time_before_cooldown_sub"
                    ]
        agents = []
        for i, cluster in enumerate(self.sub_areas):
            converter = converters.CompleteObsConverter(env, obs_attr_to_keep, cluster, self.line_areas[i], seed)
            agent = LowLevel_PPO_agent_complete_obs(env, converter, **kwargs)
            agents.append(agent)

        self.agents = np.array(agents)

    def reset(self, obs=None):
        super().reset()
        self.sub_picker.complete_reset()

    def set_training_status(self, training):
        self.training = training

    def cache_stat(self):
        cache = super().cache_stat()
        if self.goal is not None:
            (agent, action) = self.goal
            sub_vals = self.agents[agent].cache_stat()
            cache_extra = {
                "sub_vals": sub_vals,
            }
            cache.update(cache_extra)
        return cache

    def load_cache_stat(self, cache):
        super().load_cache_stat(cache)
        if self.goal is not None:
            (sub_2_act, action) = self.goal
            sub_vals = cache["sub_vals"]
            self.agents[sub_2_act].load_cache_stat(sub_vals)

    def find_cluster(self, sub_2_act):
        for i, c in enumerate(self.sub_areas):
            if sub_2_act in c:
                return i
        return -1

    def agent_act(self, obs, is_safe):
        # generate action if not safe
        if not is_safe or (len(self.sub_picker.subs_2_act) > 0):
            with torch.no_grad():
                # Middle agent chooses a substation that needs an action at this time
                sub_2_act = self.sub_picker.pick_sub(obs)
                # Select the agent that operates on the chosen substation
                agent = self.find_cluster(sub_2_act)
                idx, action, values, log_probs = self.agents[agent].act(obs, None)
                
        else:
            action = self.action_space({})
            idx = None
            values = None
            agent = None
            log_probs = None
        
        # needed to make the agent compliant with grid2op runner 
        if self.training:
            return idx, action, agent, values, log_probs
        else:
            return action

    def save_model(self, path):
        [agent.save_model(f"{path}", f"cluster_{sub}") for sub, agent in enumerate(self.agents)]

    def load_model(self, path):
        [agent.load_model(f"{path}", f"cluster_{sub}") for sub, agent in enumerate(self.agents)]