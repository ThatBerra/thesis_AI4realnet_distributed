import grid2op
from grid2op.Runner import Runner
from grid2op.Agent import RandomAgent
import multiprocessing as mp


from grid2op.Agent.agentWithConverter import AgentWithConverter
from grid2op.Converter.Converters import Converter

class SubIdToTopologyAct(Converter):

    def __init__(self, action_space):
        Converter.__init__(self, action_space)
        self.__class__ = SubIdToTopologyAct.init_grid(action_space)
        self.all_actions = []
        self.n = 1
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
        my_int = self.space_prng.randint(self.action_space.n)
        return my_int


if __name__=='__main__':
       
    env = grid2op.make("l2rpn_case14_sandbox")
    NB_EPISODE = 50 
    NB_CORE = mp.cpu_count() 
    PATH_SAVE = "agents_log"  # store the results in the "agents_log" folder

    kwargs_converter = {}
    agent = TopologyRandomAgent(env.action_space, **kwargs_converter)

    runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=agent)
    runner.run(nb_episode=NB_EPISODE, nb_process=NB_CORE, path_save=PATH_SAVE, pbar=True)