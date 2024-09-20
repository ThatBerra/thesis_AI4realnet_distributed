import torch
import itertools
import numpy as np
from grid2op.gym_compat import GymEnv, BoxGymObsSpace
from grid2op.Converter import Converter
from utils.box_gym_obsspace import MaskedBoxGymObsSpace
from utils.custom_gym_actspace import ClusterActionSpace

class ClusterConverter(Converter): 
    def __init__(self, env, obs_attr_to_keep, sub_cluster, line_cluster, seed):
        Converter.__init__(self, env.action_space)
        self.__class__ = ClusterConverter.init_grid(env.action_space)
        self.all_actions = []
        self.n = 1  # just init
        self._init_size = env.action_space.size()
        self.kwargs_init = {}
        self.sub_cluster = sub_cluster
        self.line_cluster = line_cluster

        self.create_mask(env, obs_attr_to_keep)
        
        self.init_actions(env)

        self.define_gymEnv(env, obs_attr_to_keep, seed)
        self.dim_obs = len(self.convert_obs(env.observation_space.get_empty_observation()))

    def init_actions(self, env, all_actions=None):

        for sub_id in self.sub_cluster:
            if env.action_space.sub_info[sub_id] > 3:
                self.all_actions += self.get_all_unitary_topologies_set(self, sub_id=sub_id)

        self.num_actions = len(self.all_actions)

    def convert_act(self, encoded_act):
        act_idx, values, log_probs = encoded_act
        return act_idx, self.all_actions[act_idx], values, log_probs

    def define_gymEnv(self, env, obs_attr_to_keep, seed):
        gymenv_kwargs={}
        self.gymEnv = GymEnv(env, **gymenv_kwargs)

        self.gymEnv.observation_space.close()
        obs_space_kwargs = {}   

        self.gymEnv.observation_space = MaskedBoxGymObsSpace(env.observation_space,
                                            self.mask_dict,
                                            attr_to_keep=obs_attr_to_keep,
                                            **obs_space_kwargs)
        self.gymEnv.action_space.close()
        self.gymEnv.action_space = ClusterActionSpace(self.num_actions, seed)
    
    def create_mask(self, env, obs_attr_to_keep):
        
        area_grid_objects_types = []
        for el in env.observation_space.grid_objects_types:
            if el[0] in self.sub_cluster:
                area_grid_objects_types.append(list(el))

        area_grid_objects_types = np.array(area_grid_objects_types)

        area_ids = [[], [], [], [], []]
        topo_idx = [[], [], [], [], []]

        for el in area_grid_objects_types:
            idx = np.argwhere(el >= 0)[1][0]  # retrieve the index of the non-zero element (take index 1 because there is always column 0 with the substation id)
            # based on the retrieved index you know what is the current object, so you can put its id inside the correct list (idx-1)
            area_ids[idx-1].append(el[idx])
            # get the idx of the object inside the topo vect = it is the row idx of that el inside grid_objects_types
            topo_idx[idx-1].append(np.where((env.observation_space.grid_objects_types == el).all(axis=1))[0][0])

        obs_attr = sorted(obs_attr_to_keep)
        obj_typename = ["load", "gen", "line_or", "line_ex", "stor"]
        
        full_obs_dim = dict(zip(env.observation_space.attr_list_vect, env.observation_space.shape))
        obs_dim = {}
        obs_dim = {key: full_obs_dim[key] for key in obs_attr}

        attr_by_obj_type = {
    
            'load' : ["load_p",
                "load_q",
                "load_v",
                "load_theta"],

            'gen' : ["gen_p",
                "gen_p_before_curtail",
                "gen_q",
                "gen_v",
                "gen_margin_up",
                "gen_margin_down",
                "target_dispatch",
                "actual_dispatch",
                "curtailment",
                "curtailment_limit",
                "curtailment_limit_effective",
                "thermal_limit",
                "gen_theta"],
    
            'line_or' : ["p_or",
                "q_or",
                "v_or",
                "a_or",
                "theta_or"],

            'line_ex' : ["p_ex",
                "q_ex",
                "v_ex",
                "a_ex",
                "theta_ex"],

            'stor' : ["storage_charge",
                "storage_power_target",
                "storage_power"],

            'line' : ["rho",
                "line_status",
                "timestep_overflow",
                "time_before_cooldown_line",
                "time_next_maintenance",
                "duration_next_maintenance"],

            'sub' : ["time_before_cooldown_sub"],

            'topo' : ["topo_vect"],

            'datetime' : ["year",
                "month",
                "day",
                "hour_of_day",
                "minute_of_hour",
                "day_of_week"]

            }

        area_ids_dict = dict(zip(obj_typename, [sorted(l) for l in area_ids]))
        topo_idx_dict = dict(zip(obj_typename, [sorted(l) for l in topo_idx]))
        mask = []
        self.mask_dict = {}

        for attr_name, dim in obs_dim.items():
            for obj_type, attr_list in attr_by_obj_type.items():
                if attr_name in attr_list:
                    if obj_type in obj_typename:
                        rel_idx = area_ids_dict[obj_type]
                    elif obj_type == "line":
                        #rel_idx = list(set(area_ids_dict['line_or'] + area_ids_dict['line_ex']))  # get unique line IDs
                        rel_idx = self.line_cluster                    
                    elif obj_type == 'topo':
                        rel_idx = list(set(sum(topo_idx_dict.values(), [])))  # flatten list
                    elif obj_type == 'sub':
                        rel_idx = self.sub_cluster
                    elif obj_type == 'datetime':
                        rel_idx = [0]
                    mask_tmp = np.zeros(dim, dtype='bool')
                    mask_tmp[rel_idx] = True
                    self.mask_dict[attr_name] = rel_idx
                    #print(f"{attr_name} ---> {obj_type} ---> {rel_idx}, {mask_tmp}")
                    mask.append(mask_tmp)

        self.mask = sum([list(el) for el in mask], [])
        
    
    def convert_obs(self, obs):
        gym_obs = self.gymEnv.observation_space.to_gym(obs)

        #masked_obs = gym_obs[self.mask]
        return gym_obs

class CompleteObsConverter(Converter): 
    def __init__(self, env, obs_attr_to_keep, sub_cluster, line_cluster, seed):
        Converter.__init__(self, env.action_space)
        self.__class__ = CompleteObsConverter.init_grid(env.action_space)
        self.all_actions = []
        self.n = 1  # just init
        self._init_size = env.action_space.size()
        self.kwargs_init = {}
        self.sub_cluster = sub_cluster
        self.line_cluster = line_cluster

        #self.create_mask(env, obs_attr_to_keep)
        self.define_gymEnv(env, obs_attr_to_keep, seed)
        
        self.dim_obs = len(self.convert_obs(env.observation_space.get_empty_observation()))
        self.init_converter(self)

    def init_converter(self, all_actions=None):

        for sub_id in self.sub_cluster:

            self.all_actions += self.get_all_unitary_topologies_set(self, sub_id=sub_id)

        self.num_actions = len(self.all_actions)

    def convert_act(self, encoded_act):
        act_idx, values, log_probs = encoded_act
        return act_idx, self.all_actions[act_idx], values, log_probs

    def define_gymEnv(self, env, obs_attr_to_keep, seed):
        gymenv_kwargs={}
        self.gymEnv = GymEnv(env, **gymenv_kwargs)

        self.gymEnv.observation_space.close()
        self.gymEnv.observation_space = BoxGymObsSpace(env.observation_space)

        self.gymEnv.action_space.close()
        self.gymEnv.action_space = ClusterActionSpace(self.num_actions, seed)
        
    def convert_obs(self, obs):
        gym_obs = self.gymEnv.observation_space.to_gym(obs)

        #masked_obs = gym_obs[self.mask]
        return gym_obs
