from pysc2.env import sc2_env
from pysc2.lib import features
from absl import app
from scripted_bot import TerranAgent
from data import Data_Container
from study_observation import take_snapshot


def main(unused_argv):
    agent = TerranAgent()
    storage = Data_Container()
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
                take_snapshot(timesteps[0])
                raise AssertionError
                agent.reset()
                while True:
                    step_actions = [agent.step(timesteps[0], storage)]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
