import sys
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
FLAGS(sys.argv)

map_size = 32
step_mul = 1
visualize = False

env = sc2_env.SC2Env(map_name='MoveToBeacon',
    agent_interface_format=sc2_env.parse_agent_interface_format(
        feature_screen=map_size,        # screen size
        feature_minimap=map_size,       # minimap size  
        rgb_screen=None,
        rgb_minimap=None,
        action_space=None,
        use_feature_units=False),
    step_mul=step_mul,
    game_steps_per_episode=None,
    disable_fog=False,
    visualize=visualize)                     # visualize on

obs = env.reset()

while True:
    # https://github.com/deepmind/pysc2/blob/master/pysc2/lib/actions.py#L450

    ## 0. check available_action
    print(obs[0].observation.available_actions)
    
    ## 1.
    # action = actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [[0], [10, 10]])        # not queue, [10, 10], move, actions.FUNCTIONS.Move_screen.id = 331

    ## 2.
    # action = actions.FunctionCall(actions.FUNCTIONS.select_unit.id, [[0], 317])

    ## 3.
    action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
    
    obs = env.step(actions=[action])
    reward = obs[0].reward
    done = obs[0].step_type == environment.StepType.LAST

    if done:
        obs = env.reset()