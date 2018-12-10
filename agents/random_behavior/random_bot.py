from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
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
        x_space_size = 61
        y_space_size = 81
        action_space_size = x_space_size * y_space_size * 2

        # first, pick a random number to represent some action from the action space
        rho = randint(action_space_size)
        rho_stored = rho

        # determine if action is attack (<1200) or move (>=1200)
        if rho < (action_space_size / 2):
            action = 'attack'
        else:
            action = 'move'
            rho -= (action_space_size / 2)

        # calc x and y from the random number
        x = rho // x_space_size
        y = rho % x_space_size
        target = (x, y)

        if action == 'attack':
            if self.can_do(obs, actions.FUNCTIONS.Attack_screen.id):
                return actions.FUNCTIONS.Attack_screen("now", target), rho_stored
        elif action == 'move':
            if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", target), rho_stored

        return actions.FUNCTIONS.no_op(), rho_stored
