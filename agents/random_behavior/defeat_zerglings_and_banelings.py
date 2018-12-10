from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import numpy as np
from numpy.random import randint
from random_bot import TerranAgent
from data import Data_Container


def main(unused_argv):
    agent = TerranAgent()
    storage = Data_Container()
    num_games = 100
    try:
        for i in range(num_games):
            print('\n------------------------------------------------')
            print('Game: {} / {}'.format(i + 1, num_games))
            print('------------------------------------------------\n')
            with sc2_env.SC2Env(
                map_name="DefeatZerglingsAndBanelings",
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(feature_dimensions=features.Dimensions(screen=84, minimap=64), use_feature_units=True),
                step_mul=8,
                game_steps_per_episode=0,
                visualize=False) as env:
                    agent.setup(env.observation_spec(), env.action_spec())
                    timesteps = env.reset()
                    agent.reset()
                    storage.log_step(timesteps[0], 0)
                    while True:
                        action, action_index = agent.step(timesteps[0])
                        step_actions = [action]
                        if timesteps[0].last():
                            break
                        timesteps = env.step(step_actions)
                        storage.log_step(timesteps[0], action_index)
            
    except KeyboardInterrupt:
        pass
    
    # store data
    storage.save_data('logged_data.npy')

    return
  
if __name__ == "__main__":
    app.run(main)
