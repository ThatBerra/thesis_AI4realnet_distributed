import grid2op
from grid2op.Runner import Runner
from grid2op.Agent import RandomAgent
import multiprocessing as mp
import numpy as np
import time
import csv
import os

from grid2op.Agent.agentWithConverter import AgentWithConverter
from grid2op.Converter.Converters import Converter

class SubIdToTopologyAct(Converter):

    def __init__(self, action_space):
        Converter.__init__(self, action_space)
        self.__class__ = SubIdToTopologyAct.init_grid(action_space)
        self.all_actions = []
        self.n = 1  # just init
        self._init_size = action_space.size()
        self.kwargs_init = {}

    def init_converter(self, all_actions=None, **kwargs):
        self.kwargs_init = kwargs
        self.all_actions = self.get_all_unitary_topologies_set(self, **kwargs)
        self.n = len(self.all_actions)

    def convert_act(self, encoded_act):
        return self.all_actions[encoded_act]

class TopologyRandomAgent(AgentWithConverter):
    def __init__(
        self, action_space, action_space_converter=SubIdToTopologyAct, **kwargs_converter
    ):
        AgentWithConverter.__init__(
            self, action_space, action_space_converter, **kwargs_converter
        )
        # print('Hey there')

    def my_act(self, transformed_observation, reward, done=False):
        return self.space_prng.randint(self.action_space.n)


if __name__=='__main__':
       
    env = grid2op.make("l2rpn_case14_sandbox")
    
    seed = 29
    
    NB_EPISODE = 5
    NB_CORE = mp.cpu_count() 
    PATH_SAVE = "case14_test2"

    print(f"NB_EPISODE = {NB_EPISODE}, nb_scenario = {len(env.chronics_handler.subpaths)}")

    kwargs_converter = {}
    agent = TopologyRandomAgent(env.action_space, **kwargs_converter)

    np.random.seed(seed)
    env_seeds = np.random.choice(int(1e8), size=NB_EPISODE, replace=False)
    agent_seeds = np.random.choice(int(1e8), size=NB_EPISODE, replace=False)

    runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=agent)

    st_time = time.time()
    res = runner.run(nb_episode=NB_EPISODE, nb_process=NB_CORE, path_save=PATH_SAVE, pbar=True,
                     env_seeds=env_seeds, agent_seeds=agent_seeds)
    end_time = time.time()

    with open(os.path.join(PATH_SAVE,'res.csv'), 'w') as f:
        write = csv.writer(f)
        write.writerows(res)  # (id_chron, name_chron, float(cum_reward), nb_time_step, max_ts)

    with open(os.path.join(PATH_SAVE,'time.txt'), 'w') as f:
        f.write(str(round(end_time-st_time, 2)))

    print(round(end_time-st_time, 2))