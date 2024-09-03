import torch
import os

from grid2op.Agent import AgentWithConverter

from stable_baselines3.ppo import PPO
from stable_baselines3.common.buffers import RolloutBuffer


class LowLevel_PPO_agent(AgentWithConverter):
    def __init__(self, env, action_space_converter=None, **kwargs_converter):
        super().__init__(env.action_space, action_space_converter, **kwargs_converter)
        self.gymEnv = action_space_converter.gymEnv
        self.ppo_agent = PPO('MlpPolicy', action_space_converter.gymEnv,  n_steps=16, batch_size=4)
        self._last_episode_starts = False
        self.n_updates=0

    def check_train(self, len):
        if self.ppo_agent.rollout_buffer.buffer_size >= len:
            return True
        return False

    def my_act(self, obs, reward=None, done=False):
        # generate action if not safe
        with torch.no_grad():
            obs = [obs]
            obs_tensor = torch.as_tensor(obs, device=self.ppo_agent.device)
            encoded_act, values, log_probs = self.ppo_agent.policy(obs_tensor)

        return encoded_act, values, log_probs
    
    
    def update_buffer(self, observation, action, reward, _last_episode_starts, values, log_probs,):
        obs = self.action_space_converter.convert_obs(observation)
        self.ppo_agent.rollout_buffer.add(
                    obs,  # type: ignore[arg-type]
                    action,
                    reward,
                    _last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs,
                )
        
    def save_model(self, path, name):
        save_path = os.path.join(path, name)
        self.ppo_agent.save(save_path)
        
    def load_model(self, path, name):
        load_path = os.path.join(path, name)
        self.ppo_agent = self.ppo_agent.load(load_path, env=self.gymEnv)


class LowLevel_PPO_agent_complete_obs(AgentWithConverter):
    def __init__(self, env, action_space_converter=None, **kwargs_converter):
        super().__init__(env.action_space, action_space_converter, **kwargs_converter)
        self.gymEnv = action_space_converter.gymEnv
        self.ppo_agent = PPO('MlpPolicy', action_space_converter.gymEnv, n_steps=16, batch_size=4)
        self._last_episode_starts = False
        self.n_updates=0

    def check_train(self, len):
        if self.ppo_agent.rollout_buffer.buffer_size >= len:
            return True
        return False

    def my_act(self, obs, reward=None, done=False):
        # generate action if not safe
        with torch.no_grad():
            obs = [obs]
            obs_tensor = torch.as_tensor(obs, device=self.ppo_agent.device)
            encoded_act, values, log_probs = self.ppo_agent.policy(obs_tensor)

        return encoded_act, values, log_probs
    
    
    def update_buffer(self, observation, action, reward, _last_episode_starts, values, log_probs,):
        obs = self.action_space_converter.convert_obs(observation)
        self.ppo_agent.rollout_buffer.add(
                    obs,  # type: ignore[arg-type]
                    action,
                    reward,
                    _last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs,
                )
        
    def save_model(self, path, name):
        save_path = os.path.join(path, name)
        self.ppo_agent.save(save_path)
        
    def load_model(self, path, name):
        load_path = os.path.join(path, name)
        self.ppo_agent = self.ppo_agent.load(load_path, env=self.gymEnv)