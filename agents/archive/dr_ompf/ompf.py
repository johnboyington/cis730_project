from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import numpy as np


class TerranAgent(base_agent.BaseAgent):
    
    def __init__(self):
        super(TerranAgent, self).__init__()
        self.friendly_unit_list = []
        self.unit_index = 0
        self.action_step = 'DESELECT'
        
        # hardcoded parameters for defeatroaches scenario
        self.enemy_range = 4
        self.friendly_range = 5

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
                    return actions.FUNCTIONS.select_point("toggle", (unit.x, unit.y))
        else:
            self.action_step = 'SELECT'
            return actions.FUNCTIONS.no_op()
    
    def select_next_marine(self, marines):
        """Utility that selects the next marine in the queue."""
        if self.unit_index >= len(marines):
            self.unit_index = 0
        unit = marines[self.unit_index]
        self.current_unit = unit
        self.unit_index += 1
        self.action_step = 'ACT'
        return actions.FUNCTIONS.select_point("select", (unit.x, unit.y))
    
    def act(self, obs):
        """Act."""
        # change action step
        self.action_step = 'DESELECT'
        
        # get additional inferred data
        self.inferred_observations(obs)
        
        # get priority attack queue
        target_priority_queue = self.calc_target_priority_queue(self.available_targets)
        
        # find the best position
        best_position = self.calc_best_position(target_priority_queue)
        
        # move to best position if possible
        if self.can_do(obs, actions.FUNCTIONS.Move_screen.id):
            return actions.FUNCTIONS.Move_screen("now", best_position)
        else:
            return actions.FUNCTIONS.no_op()
    
    def inferred_observations(self, obs):
        """Calculates information inferred from observation data."""
        
        self.available_targets = self.get_units_by_type(obs, units.Zerg.Roach)
        # calc center of mass of targets
        xx = 0
        yy = 0
        for target in self.available_targets:
            xx += target.x
            yy += target.y
        self.enemy_xcom = xx / len(self.available_targets)
        self.enemy_ycom = yy / len(self.available_targets)
        return
        
    
    def calc_target_priority_queue(self, available_targets):
        """Returns ordered list of priority targets."""
        
        # the following function helps sort targets
        def priority_function(target):
            # A and B are parameters that help determine target's priority score
            A = 1
            B = -.2
            dist_from_com = np.sqrt((target.x - self.enemy_xcom)**2 + (target.y - self.enemy_ycom)**2)
            return (A * target.health) + (B * dist_from_com)
        
        # sort targets based on priority
        return sorted(available_targets, key=priority_function)
    
    def calc_best_position(self, target_priority_queue):
        """Calculates a list of best move locations."""
        
        # calc x and y positions
        #ord_lim = 40
        #xx = np.linspace(self.enemy_xcom - ord_lim, self.enemy_xcom + ord_lim, 80)
        #yy = np.linspace(self.enemy_ycom - ord_lim, self.enemy_ycom + ord_lim, 80)
        
        xx = np.linspace(0, 120, 121)
        yy = np.linspace(0, 100, 101)
        
        # loop through each point
        points = []
        for x in xx:
            for y in yy:
                # calculate the score here
                # first calc number of enemies that can fire on square
                N_E = 0
                for enemy in self.available_targets:
                    if np.sqrt((x - enemy.x)**2 + (y - enemy.y)**2) < self.enemy_range:
                        N_E += 1
                
                # is the highest priority target in range?
                hpt = target_priority_queue[0]
                D_P = 0
                if np.sqrt((x - hpt.x)**2 + (y - hpt.y)**2) < self.friendly_range:
                    D_P = 1
                
                # distance from current position
                D_C_POS = np.sqrt((x - self.current_unit.x)**2 + (y - self.current_unit.y)**2)
                
                
                # calculate score
                weights = np.array([1, -2, 0.0])
                score_categories = np.array([N_E, D_P, D_C_POS])
                score = np.sum(weights * score_categories)
                
                # if not in range, remove score from possible move options
                if D_P == 0:
                    score = 100
                
                points.append(((x, y), score))
                
        # return the points sorted by score
        points = sorted(points, key=lambda x: x[1])
        return points[0][0]
    
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
            return self.act(obs)
        
        
        return actions.FUNCTIONS.no_op()


def main(unused_argv):
    agent = TerranAgent()
    try:
        with sc2_env.SC2Env(
            map_name="DefeatRoaches",
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(feature_dimensions=features.Dimensions(screen=124, minimap=64), use_feature_units=True),
            step_mul=1,
            game_steps_per_episode=0,
            visualize=True) as env:
                agent.setup(env.observation_spec(), env.action_spec())
                timesteps = env.reset()
                agent.reset()
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)
    except KeyboardInterrupt:
        pass
  
if __name__ == "__main__":
    app.run(main)