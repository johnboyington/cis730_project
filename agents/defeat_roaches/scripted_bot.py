from pysc2.agents import base_agent
from pysc2.lib import actions, units
from transform_action import transform_action


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

    def step(self, obs, storage):
        super(TerranAgent, self).step(obs)

        # get list of all visible roaches
        roaches = self.get_units_by_type(obs, units.Zerg.Roach)
        roaches = sorted(roaches, key=lambda x: (x.health, x.y))
        target = roaches[0].x, roaches[0].y

        # determine a label for the action and log data
        action_id = transform_action(target)
        storage.log_step(obs, action_id)

        # attack target with marines
        if self.can_do(obs, actions.FUNCTIONS.Attack_screen.id):
            return actions.FUNCTIONS.Attack_screen("now", target)

        return actions.FUNCTIONS.no_op()
