import os
import random
import numpy as np
import torch
import grid2op
import shutil

from lightsim2grid import LightSimBackend
from grid2op.Reward import CloseToOverflowReward
from grid2op.Agent import DoNothingAgent

from Agents.HigLevel import IMARL, IMARL_complete_obs
from utils.train import Trainer


SEED = 90566

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def overflow(obs):
    for c in obs.rho:
        if c >= 1:
            return True
    return False

def safe(obs):
    for c in obs.rho:
        if c >= 0.9:
            return False
    return True

def collect_baseline_data(env, actor, nb_episode):
    safes = []
    overflows = []
    survived = []
    rewards = []

    path = f'baseline_data_{nb_episode}_episodes'

    env.seed(SEED)
    obs = env.reset()
    for episode in range(nb_episode):
        first_unsafe = False
        print(f'episode: {episode+1}')
        done = False
        ep_safe = 0
        #ep_overflow = 0
        ep_survived = 0
        ep_reward = 0

        reward = 0
        while not done:
            if safe(obs) and not first_unsafe:
                ep_safe += 1
            else: 
                first_unsafe = True 

            act = actor.act(obs, reward)
            obs, reward, done, info = env.step(act)
            ep_reward += reward

            if done:
                print(f'survived {ep_survived} timesteps, safe for {ep_safe}')
                print('-------')
                survived.append(ep_survived)
                safes.append(ep_safe)
                rewards.append(ep_reward)
                obs = env.reset()
            else:
                ep_survived += 1

    print(f'{nb_episode} episodes done, saving')
    print(len(survived), len(overflows), len(safes), len(rewards))
    os.makedirs(path, exist_ok=True)

    survived_path = os.path.join(path, 'survived_timesteps.npy')
    safes_path = os.path.join(path, 'safes.npy')
    reward_path = os.path.join(path, 'rewards.npy')

    with open(survived_path, 'wb') as f:
        np.save(f, survived)

    with open(safes_path, 'wb') as f:
        np.save(f, safes)

    with open(reward_path, 'wb') as f:
        np.save(f, rewards)

    print('saved_data')
    return

if __name__ == "__main__":
    #Inputs: modify as will
    env_name = 'l2rpn_case14_sandbox'
    
    #clusters for the decomposed problem
    sub_clusters = [
            [0, 1, 2, 4],
            [3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            ]
    
    line_clusters = [
        [0,1,2,3,4,5,6],
        [7,8,9,10,11,12,13,14,15,16,17,18,19]
    ]
    '''Uncomment for centralized model
    sub_clusters = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            ]

    line_clusters = [
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    ]'''
    
    # number of iterations over the training dataset
    n_iterations = 25

    # --------------------------------------------------------------------
    seed_everything(SEED)

    if torch.cuda.is_available():
        print(">> >> using cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # This will generate the train-validation split in "data_grid2op" folder, execute once the first time
    # Any new execution will erase data collected in that folder (for example with an EpisodeStatistic run)
    '''try:
        nm_env_train, nm_env_test = env.train_val_split_random(pct_val=10.)
    except Exception as e: 
        shutil.rmtree("C:\\Users\\david\\data_grid2op\\l2rpn_case14_sandbox_train")
        shutil.rmtree("C:\\Users\\david\\data_grid2op\\l2rpn_case14_sandbox_val")
        nm_env_train, nm_env_test = env.train_val_split_random(pct_val=10.)'''
    
    # Choose the type of environment you want to create, remember that LightSimBackend is faster, 
    # but it does not work on all machines
    
    env_train = grid2op.make(env_name+"_train", reward_class=CloseToOverflowReward, backend=LightSimBackend())
    #env_train = grid2op.make(env_name+"_train", reward_class=CloseToOverflowReward)
    
    nb_scenario = n_iterations * len(env_train.chronics_handler.subpaths)
    
    my_agent = IMARL(env_train, sub_clusters, line_clusters, SEED, **{})

    '''Uncomment if you want to load a particular model from a checkpoint

    checkpoint_path = 'distributed_PPO_47008episodes_64\\11752'
    my_agent.load_model(checkpoint_path)'''

    trainer = Trainer(env_train, my_agent, nb_scenario, SEED)
    trainer.learn(nb_scenario)






