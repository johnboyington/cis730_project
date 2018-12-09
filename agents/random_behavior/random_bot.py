from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import numpy as np
from numpy.random import randint


class TerranAgent(base_agent.BaseAgent):
    
    def __init__(self):
        super(TerranAgent, self).__init__()

    
    def can_do(self, obs, action):
        return action in obs.observation.available_actions


    def step(self, obs):
        """This agent will either attack or move to a random square."""
        super(TerranAgent, self).step(obs)
        
        # establish some ground rules
        action_space_size = 24442
        x_space_size = 101
        y_space_size = 121
        
        # first, pick a random number to represent some action from the action space
        rho = randint(x_space_size * y_space_size * 2)
        
        
        # determine if action is attack (<1200) or move (>=1200)
        if rho < (action_space_size / 2):
            action = 'attack'
        else:
            action = 'move'
            rho -= (action_space_size / 2)
        
        # calc x and y from the random number
        y = rho // x_space_size
        x = rho % x_space_size
        target = (x, y)
        
        print(action, target)

        if action == 'attack':
            if self.can_do(obs, actions.FUNCTIONS.Attack_screen.id):
                return actions.FUNCTIONS.Attack_screen("now", target)
        elif action == 'move':
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", target)
        
        
        return actions.FUNCTIONS.no_op()

