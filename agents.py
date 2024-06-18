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