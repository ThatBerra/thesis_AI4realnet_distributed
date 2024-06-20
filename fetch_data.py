import numpy as np
import grid2op
from grid2op.Episode import EpisodeData
import os
import time
from tqdm import tqdm

if __name__=='__main__':


    path = "./wcci2022_30k"
    # env = grid2op.make("l2rpn_case14_sandbox")
    env = grid2op.make("l2rpn_wcci_2022")

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

    list_dir = os.listdir(path)[:20]
    num_folders = len(list_dir)

    st_time = time.time()
    print(f"\nPath = {path}. Reading data from {num_folders} folders")

    with tqdm(total=num_folders) as pbar:
        for dire in list_dir:

            d_path = os.path.join(path, dire)
            if os.path.isdir(d_path):
                o_path = os.path.join(d_path, 'observations.npz')
                a_path = os.path.join(d_path, 'actions.npz')

                try:
                    obs = np.load(o_path)['data']
                    obs = obs[~np.isnan(obs).all(axis=1)]
                    obs = obs[:, o_start_idx:o_end_idx]
                    curr_state.extend(obs[:-1])
                    next_state.extend(obs[1:])

                    acts = np.load(a_path)['data']
                    acts = acts[~np.isnan(acts).all(axis=1)]
                    acts = acts[:, a_start_idx:a_end_idx]
                    actions.extend(acts)
                except Exception as e:
                    print(e)
            pbar.update(1)

    end_time = time.time()
    print(f"Elapsed time: {round(end_time-st_time,2)}\n")


    # ------------------------------------------------------------
    # Actions by substations

    n_sub = len(connections)
    actions = np.array(actions)
    H = actions.shape[0]

    actions_by_sub = np.zeros((H,n_sub))

    print("\nCreating actions by substation")
    st_time = time.time()

    for sub_id in range(n_sub):
        sub_act = actions[:, sub_start[sub_id]:sub_end[sub_id]]
        u_act, act_to_int = np.unique(sub_act, axis=0, return_inverse=True)
        actions_by_sub[:, sub_id] = act_to_int/u_act.shape[0]+1

    end_time = time.time()
    print(f"Elapsed time: {round(end_time-st_time,2)}\n")


    # ------------------------------------------------------------
    # Save history npz

    print("\nSaving compressed history")
    st_time = time.time()

    next_state = np.array(next_state)
    curr_state = np.array(curr_state)

    history = np.append(next_state, curr_state, axis=1)
    history = np.append(history, actions_by_sub,axis=1)

    print(history.shape)

    np.savez_compressed(f'{path}_hist.npz', data=history)

    end_time = time.time()
    print(f"Elapsed time: {round(end_time-st_time,2)}\n")

    


            

    
    