import numpy as np
import grid2op
from grid2op.Episode import EpisodeData
import os

if __name__=='__main__':


    path = "./case14_test"
    env = grid2op.make("l2rpn_case14_sandbox")
    #env = grid2op.make("l2rpn_wcci_2022")

    full_obs_dim = dict(zip(env.observation_space.attr_list_vect, env.observation_space.shape))

    o_start_idx = 0
    o_end_idx = 0
    for attr_name, dim in full_obs_dim.items():
        o_end_idx = o_start_idx + dim
    
        if attr_name == 'rho':
            break
        else:
            o_start_idx = o_end_idx
    #######################
    full_acts_dim = dict(zip(env.action_space.attr_list_vect, env.action_space.shape))

    a_start_idx = 0
    a_end_idx = 0
    for attr_name, dim in full_acts_dim.items():
        a_end_idx = a_start_idx + dim
    
        if attr_name == '_set_topo_vect':
            break
        else:
            a_start_idx = a_end_idx
    #######################      
    connections = env.action_space.sub_info

    sub_start = []
    sub_end = []
    start = 0
    end = 0
    for n in connections:
        sub_start.append(start)
        end = start + n
        sub_end.append(end)
        start = end  
    #######################

    next_state = []
    curr_state = []
    actions = []
    for dire in os.listdir(path):

        d_path = os.path.join(path, dire)
        if os.path.isdir(d_path):
            o_path = os.path.join(d_path, 'observations.npz')
            a_path = os.path.join(d_path, 'actions.npz')

            if dire != '0225':
                obs = np.load(o_path)['data']
                obs = obs[~np.isnan(obs).all(axis=1)]
                obs = obs[:, o_start_idx:o_end_idx]
                curr_state.extend(obs[:-1])
                next_state.extend(obs[1:])

                acts = np.load(a_path)['data']
                acts = acts[~np.isnan(acts).all(axis=1)]
                acts = acts[:, a_start_idx:a_end_idx]
                actions.extend(acts)

    n_sub = len(connections)
    actions = np.array(actions)
    H = actions.shape[0]

    actions_by_sub = np.zeros((H,n_sub))

    for sub_id in range(n_sub):
        sub_act = actions[:, sub_start[sub_id]:sub_end[sub_id]]
        u_act, act_to_int = np.unique(sub_act, axis=0, return_inverse=True)
        actions_by_sub[:, sub_id] = act_to_int/u_act.shape[0]+1

    next_state = np.array(next_state)
    curr_state = np.array(curr_state)

    history = np.append(next_state, curr_state, axis=1)
    history = np.append(history, actions_by_sub,axis=1)

    print(history.shape)

    


            

    
    