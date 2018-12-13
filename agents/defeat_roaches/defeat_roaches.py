from pysc2.env import sc2_env
from pysc2.lib import features
from absl import app
import sys
from scripted_bot import TerranAgent as Logger
from learning_bot import TerranAgent as Learner
from data import Data_Container

assert sys.argv[1] in ['log', 'learn']

def main(unused_argv):
    if sys.argv[1] == 'log':
        agent = Logger()
        storage = Data_Container(method='compress')
        visual = False
        num_runs = 10
    elif sys.argv[1] == 'learn':
        agent = Learner()
        storage = None
        visual = True
        num_runs = 1
    try:
        for i in range(num_runs):
            print('--------------------------------- GAME {} / {}'.format(i+1, num_runs))
            with sc2_env.SC2Env(
                map_name="DefeatRoaches",
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(feature_dimensions=features.Dimensions(screen=124, minimap=64), use_feature_units=True),
                step_mul=1,
                game_steps_per_episode=0,
                visualize=visual) as env:
                    agent.setup(env.observation_spec(), env.action_spec())
                    timesteps = env.reset()
                    agent.reset()
                    while True:
                        step_actions = [agent.step(timesteps[0], storage)]
                        if timesteps[0].last():
                            break
                        timesteps = env.step(step_actions)
    except KeyboardInterrupt:
        pass
    if sys.argv[1] == 'log':
        storage.save_data()


if __name__ == "__main__":
    app.run(main)
