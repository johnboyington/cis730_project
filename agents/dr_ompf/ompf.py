from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import numpy as np


class TerranAgent(base_agent.BaseAgent):
    
    def __init__(self):
        super(TerranAgent, self).__init__()
        self.attack_coordinates = None

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit_type):
            return True
        if (len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == unit_type):
            return True
        return False
    

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    
    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    
    def calc_target_priority_queue(self, available_targets):
        """Returns ordered list of priority targets."""
        
        # calc center of mass of targets
        xx = 0
        yy = 0
        for target in available_targets:
            xx += target.x
            yy += target.y
        self.enemy_xcom = xx / len(available_targets)
        self.enemy_ycom = yy / len(available_targets)
        
        # the following function helps sort targets
        def priority_function(target):
            # A and B are parameters that help determine target's priority score
            A = 1
            B = -.2
            dist_from_com = np.sqrt((target.x - self.enemy_xcom)**2 + (target.y - self.enemy_ycom)**2)
            return (A * target.health) + (B * dist_from_com)
        
        # sort targets based on priority
        return sorted(available_targets, key=priority_function)
    
    def step(self, obs):
        super(TerranAgent, self).step(obs)
        
        # grab one marine
        if not self.unit_type_is_selected(obs, units.Terran.Marine):
            marines = self.get_units_by_type(obs, units.Terran.Marine)
            if marines:
                return actions.FUNCTIONS.select_point("select", (marines[0].x, marines[0].y))
            else:
                pass
        
        # get target priority queue
        roaches = self.get_units_by_type(obs, units.Zerg.Roach)
        target_priority_queue = self.calc_target_priority_queue(roaches)
        
        # can agent attack any of these priority targets?
        for target in target_priority_queue:
            if self.can_do(obs, actions.FUNCTIONS.Attack_screen.id):
                return actions.FUNCTIONS.Attack_screen("now", (target.x, target.y))
        
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