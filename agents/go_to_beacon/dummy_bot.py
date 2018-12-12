from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
import numpy as np


class TerranAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranAgent, self).__init__()
        self.first = True

    def get_units_by_type(self, obs, unit_type):
        units = [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]
        return sorted(units, key=lambda u: (u.x, u.y))
    
    def step(self, obs):
        """This agent will either attack or move to a random square."""
        super(TerranAgent, self).step(obs)
        
        # if the first step, select the marine
        if self.first:
            self.first = False
            marines = self.get_units_by_type(obs, units.Terran.Marine)
            unit = marines[0]
            return actions.FUNCTIONS.select_point("select", (unit.x, unit.y))
        
        # print stuff from observation
        fs = obs.observation.feature_screen
        for row in fs.player_id:
            print(row)

        # move to a random location
        x = np.random.randint(80)
        y = np.random.randint(80)
        
        return actions.FUNCTIONS.Move_screen("now", (x, y))
