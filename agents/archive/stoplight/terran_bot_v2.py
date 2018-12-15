from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import numpy as np
from numpy.random import randint


class TerranAgent(base_agent.BaseAgent):
    
    def __init__(self):
        super(TerranAgent, self).__init__()
        self.friendly_unit_list = []
        self.unit_index = 0
        self.action_step = 'DESELECT'
        self.distance_between = self.calc_distance_grid()

    
    def calc_distance_grid(self):
        """Stores the distance between any two points to speed up calculations"""
        distance_between = np.zeros((120, 120))
        for i in range(len(distance_between)):
            for j in range(len(distance_between)):
                distance_between[i, j] = np.sqrt(i**2 + j**2)
        return distance_between
    
    def calc_distance_between(self, x1, y1, x2, y2):
        """Calculates the distance between any two sets of x, y points using the precalculated distance grid."""
        return self.distance_between[int(abs(x1 - x2)), int(abs(y1 - y2))]


    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit_type):
            return True
        if (len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == unit_type):
            return True
        return False
    

    def get_units_by_type(self, obs, unit_type):
        units = [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]
        return sorted(units, key=lambda u: (u.x, u.y))

    
    def can_do(self, obs, action):
        return action in obs.observation.available_actions


    def deselect_all_units(self, obs):
        """Utility that deselects all units."""
        if self.unit_type_is_selected(obs, units.Terran.Marine):
            for unit in obs.observation.feature_units:
                if unit.is_selected:
                    print(unit)
                    return actions.FUNCTIONS.select_point("toggle", (unit.x, unit.y))
        else:
            self.action_step = 'SELECT'
            return actions.FUNCTIONS.no_op()
    
    def select_next_marine(self, marines):
        """Utility that selects the next marine in the queue."""
        self.unit_index += 1
        
        # adding this here to always select the first unit. this will be reverted later
        #self.unit_index = 0
        
        if self.unit_index >= len(marines):
            self.unit_index = 0
        unit = marines[self.unit_index]
        self.action_step = 'ACT'
        return actions.FUNCTIONS.select_point("select", (unit.x, unit.y))
    
    def act(self, obs, marines):
        # get info on current actor
        actor = marines[self.unit_index]
        
        best_position = (actor.x - 2, actor.y)
        self.action_step = 'DESELECT'
        if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
            return actions.FUNCTIONS.Move_screen("now", best_position)
        else:
            return actions.FUNCTIONS.no_op()

    
    def step(self, obs):
        super(TerranAgent, self).step(obs)
        
        # deselect something
        marines = self.get_units_by_type(obs, units.Terran.Marine)
        
        # deselect everyone
        if self.action_step == 'DESELECT':
            return self.deselect_all_units(obs)
        
        if self.action_step == 'SELECT':
            return self.select_next_marine(marines)
        
        if self.action_step == 'ACT':
            return self.act(obs, marines)
        
        
        return actions.FUNCTIONS.no_op()

