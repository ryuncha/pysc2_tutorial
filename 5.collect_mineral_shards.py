import sys
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np

FLAGS = flags.FLAGS
FLAGS(sys.argv)

map_size = 84
step_mul = 1
visualize = False

env = sc2_env.SC2Env(map_name='CollectMineralShards',
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
score = 0
first = True

while True:
    screen_player_relative = obs[0].observation.feature_screen.base[5]
    if first:
        mineral_y, mineral_x = (screen_player_relative == 3).nonzero()
        action_list = [
            actions.FunctionCall(actions.FUNCTIONS.select_rect.id, [[0], [0, 0], [map_size-1, map_size-1]]),
            actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [[0], [mineral_x[0], mineral_y[0]]])
        ]
        first = False

    else:
        mineral_y, mineral_x = (screen_player_relative == 3).nonzero()
        action_list = [
            actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [[0], [mineral_x[0], mineral_y[0]]])
        ]

    for action in action_list:
        obs = env.step(actions=[action])
        score += obs[0].reward
        done = obs[0].step_type == environment.StepType.LAST
        if done:
            break

    if done:
        print(score)
        obs = env.reset()
        score = 0
        first = True