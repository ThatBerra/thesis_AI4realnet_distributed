import os
import csv
import numpy as np
import torch
import time
import sys
import pickle

from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import datetime
from collections import deque

from stable_baselines3.common import utils
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.logger import Logger

from Agents.HigLevel import IMARL


class Trainer():
    def __init__(self, env, agent, n_episodes, seed, use_sde=False):
        self.env = env
        self.high_agent = agent
        self.n_episodes = n_episodes
        self.use_sde = use_sde
        #self.n_samples = n_samples
        self.ep_info_buffer = None  
        self.ep_success_buffer = None
        self._stats_window_size = 100
        self._num_timesteps_at_start = 0
        self.verbose = 0
        self.tensorboard_log = None
        self.episode_rewards = []
        self.episode_survived = []
        self.episode_safes = []
        self.agent_rewards = [[] for _ in range(len(self.high_agent.agents))]
        self.reset(seed)
    
    def reset(self, seed):
        self.env.seed(seed)
        self._last_obs = self.env.reset()
        self.num_timesteps = 0
        self.ep_info_buffer = deque(maxlen=self._stats_window_size)
        self.ep_success_buffer = deque(maxlen=self._stats_window_size)
        self.logger = utils.configure_logger(self.verbose, self.tensorboard_log, "run", True)

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def save_agent(self, path):
        self.high_agent.save_model(path)


    def overflow(self, obs):
        for ratio in obs.rho:
            if ratio >= 1:
                return True
        return False
    
    def safe(self, obs):
        for ratio in obs.rho:
            if ratio >= 0.9:
                return False
        return True

    def collect_episode(self, env, callbacks):
        safe = 0
        first_unsafe = False
        survived = 1
        episode_reward = 0
        agents_rewards = np.zeros(len(self.high_agent.agents))
        done = False

        for low_agent in self.high_agent.agents:
            if self.use_sde:
                low_agent.ppo_agent.policy.reset_noise(env.num_envs)
            low_agent.ppo_agent.policy.set_training_mode(False)

        for cb in callbacks:
            cb.on_rollout_start()

        self.high_agent.reset()
        
        while not done:
            obs = self._last_obs

            if self.safe(obs) and not first_unsafe:
                safe += 1
            else:
                first_unsafe = True

            act_idx, action, cluster, values, log_probs = self.high_agent.act(obs)

            new_obs, reward, done, info = env.step(action)

            if cluster is not None:
                agent = self.high_agent.agents[cluster]
                ppo_agent = agent.ppo_agent
                buff = ppo_agent.rollout_buffer
                callback = callbacks[cluster]
                
                callback.update_locals(locals())
                if not callback.on_step():
                    return False

                ppo_agent._update_info_buffer([info], done)

                buff = ppo_agent.rollout_buffer
                if buff.full:
                    with torch.no_grad():
                        terminal_obs = np.array([agent.action_space_converter.convert_obs(new_obs)])
                        terminal_value = ppo_agent.policy.predict_values(obs_as_tensor(terminal_obs, ppo_agent.device))  # type: ignore[arg-type]
                    terminal_value = terminal_value.numpy()[0][0]
                    reward += ppo_agent.gamma * terminal_value

                agents_rewards[cluster] += reward
                agent.update_buffer(
                    self._last_obs,  # type: ignore[arg-type]
                    act_idx,
                    reward,
                    agent._last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs,
                )
                agent._last_episode_starts = done
                
                if buff.full:
                    print(f"updating agent {cluster}...")
                    agent.n_updates += 1
                    with torch.no_grad():
                        # Compute value for the last timestep
                        last_obs = np.array([agent.action_space_converter.convert_obs(new_obs)])
                        values = ppo_agent.policy.predict_values(obs_as_tensor(last_obs, ppo_agent.device))  # type: ignore[arg-type]
                        buff.compute_returns_and_advantage(last_values=values, dones=done)
                        callback.update_locals(locals())
                        callback.on_rollout_end()

                    agent.ppo_agent.train()
                    agent.ppo_agent.rollout_buffer.reset()

            episode_reward += reward
            if done:
                self._last_obs = env.reset()
                self.episode_survived.append(survived)
                self.episode_safes.append(safe)
                self.episode_rewards.append(episode_reward)
                for i in range(len(self.high_agent.agents)):
                    self.agent_rewards[i].append(agents_rewards[i])
                print(f'Episode terminated -> survived for {survived} steps, safe for {safe} steps, total reward: {episode_reward}')
                print('-----')
            else:
                self._last_obs = new_obs  # type: ignore[assignment]
                survived += 1

        return True
            
    def learn(self, nb_scenario, callback=None, reset_num_timesteps=True, tb_log_name="OnPolicyAlgorithm", progress_bar=False):
        tic = time.time()
        agent_path = f'data/agents/{self.high_agent.r_seed}/{self.high_agent.agent_type}'
        data_path = f'data/training/{self.high_agent.r_seed}/{self.high_agent.agent_type}'
        
        callbacks = []
        buffers = []
        self.start_time = time.time()
        for low_agent in self.high_agent.agents:
            _, cb = low_agent.ppo_agent._setup_learn(
                nb_scenario,
                callback,
                reset_num_timesteps,
                tb_log_name,
                progress_bar,
            )
            cb.on_training_start(locals(), globals())
            callbacks.append(cb)
            self.high_agent.set_training_status(True)
            buffers.append(low_agent.ppo_agent.rollout_buffer)

        
        print(f"starting training on {nb_scenario} episodes")
        for timestep in range(nb_scenario):
            print(f'episode {timestep+1} of {nb_scenario}')
            continue_training = self.collect_episode(self.env, callbacks)

            if not continue_training:
                break
            self.num_timesteps += 1

            self._update_current_progress_remaining(timestep, nb_scenario)
            
            # Display training infos
            '''if log_interval is not None and n_episodes % log_interval == 0:
                self._dump_logs(n_episodes)'''
            
            if timestep % 904 == 0:  #TODO: Do not use number here. Use variable instead
                path = os.path.join(data_path, f'{timestep}')
                self.save_data(path, tic)

                a_path = os.path.join(agent_path, f'{timestep}')
                self.save_agent(a_path)
            
        path = os.path.join(data_path, 'final')
        self.save_data(path, tic)

        a_path = os.path.join(agent_path, 'final')
        self.save_agent(a_path)
        
        for i, agent in enumerate(self.high_agent.agents):
            print(f'agent {i} updated {agent.n_updates} times')

        toc = time.time()
        print('---------------------')
        print(f'TRAINING of {nb_scenario} episodes terminated')
        print(f'Elapsed time: {round(toc-tic,2)}')
        
            
    def save_data(self, path, tic):
        survived_path = os.path.join(path, 'survived_timesteps.npy')
        safes_path = os.path.join(path, 'safes.npy')
        agents_reward_path = os.path.join(path, 'agent_rewards.npy')
        reward_path = os.path.join(path, 'rewards.npy')

        os.makedirs(path, exist_ok=True)

        with open(survived_path, 'wb') as f:
            np.save(f, self.episode_survived)

        with open(safes_path, 'wb') as f:
            np.save(f, self.episode_safes)

        with open(agents_reward_path, 'wb') as f:
            np.save(f, self.agent_rewards)

        with open(reward_path, 'wb') as f:
            np.save(f, self.episode_rewards)

        toc = time.time()
        elapsed = round(toc-tic,2)
        t_path = os.path.join(path, 'time.txt')
        with open(t_path, 'w') as f:
            f.write(str(elapsed))


        


        