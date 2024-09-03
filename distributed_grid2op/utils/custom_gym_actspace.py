from typing import Any, Sequence
import grid2op
from gymnasium.spaces import Discrete
from matplotlib.pylab import Generator
import random

from numpy import integer


class ClusterActionSpace(Discrete):
    def __init__(self, num_actions, seed):
        n = num_actions
        self.act_var = range(num_actions)
        super().__init__(n, seed)

    def sample(self) -> Any:
        return super().sample(self.act_var)
    
    def close(self):
        return

