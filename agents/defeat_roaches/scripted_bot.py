from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app


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

    def step(self, obs):
        super(TerranAgent, self).step(obs)

        attacking = False

        # select all marines
        if not self.unit_type_is_selected(obs, units.Terran.Marine):
            marines = self.get_units_by_type(obs, units.Terran.Marine)
            if marines:
                return actions.FUNCTIONS.select_point("select_all_type", (marines[0].x, marines[0].y))
            else:
                pass

        elif not attacking:
            # get list of all visible roaches
            roaches = self.get_units_by_type(obs, units.Zerg.Roach)
            roaches = sorted(roaches, key=lambda x: (x.health, x.y))
            target = roaches[0]

            # attack target with marines
            if self.can_do(obs, actions.FUNCTIONS.Attack_screen.id):
                attacking = True
                return actions.FUNCTIONS.Attack_screen("now", (target.x, target.y))

        return actions.FUNCTIONS.no_op()
