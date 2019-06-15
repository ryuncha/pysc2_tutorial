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
    minimap_height_map = obs[0].observation.feature_minimap.base[0]         # 미니맵 고도
    minimap_visibility_map = obs[0].observation.feature_minimap.base[1]     # 미니맵 시야
    minimap_creep = obs[0].observation.feature_minimap.base[2]              # 미니맵 크립(저그)
    minimap_camera = obs[0].observation.feature_minimap.base[3]             # 미니맵 카메라
    minimap_player_id = obs[0].observation.feature_minimap.base[4]          # 미니맵 플레이어 id
    minimap_player_relative = obs[0].observation.feature_minimap.base[5]    # 미니맵 플레이어 상대적
    minimap_selected = obs[0].observation.feature_minimap.base[6]           # 미니맵 선택한 유닛

    screen_height_map = obs[0].observation.feature_screen.base[0]           # 스크린 고도
    screen_visibility_map = obs[0].observation.feature_screen.base[1]       # 스크린 시야
    screen_creep = obs[0].observation.feature_screen.base[2]                # 스크린 크립(저그)
    screen_power = obs[0].observation.feature_screen.base[3]
    screen_player_id = obs[0].observation.feature_screen.base[4]            # 미니맵 플레이어 id        
    screen_player_relative = obs[0].observation.feature_screen.base[5]      # 미니맵 플레이어 상대적
    screen_unit_type = obs[0].observation.feature_screen.base[6]            # 미니맵 유닛 타입
    screen_selected = obs[0].observation.feature_screen.base[7]             # 미니맵 선택한 유닛
    screen_unit_hit_point = obs[0].observation.feature_screen.base[8]       # 미니맵 유닛 체력
    screen_unit_hit_points_ratio = obs[0].observation.feature_screen.base[9]# 미니맵 유닛 체력 0~1
    screen_unit_energy = obs[0].observation.feature_screen.base[10]         # 미니맵 유닛 에너지
    screen_unit_energy_ratio = obs[0].observation.feature_screen.base[11]   # 미니맵 유닛 에너지 0~1
    screen_unit_shield = obs[0].observation.feature_screen.base[12]         # 미니맵 유닛 쉴드(프로토스 유닛 쉴드)
    screen_unit_shield_ratio = obs[0].observation.feature_screen.base[13]   # 미니맵 유닛 쉴드 0~1(프로토스 유닛 쉴드)
    screen_unit_density = obs[0].observation.feature_screen.base[14]
    screen_unit_density_aa = obs[0].observation.feature_screen.base[15]
    screen_effects = obs[0].observation.feature_screen.base[16]

    plt.subplot(5,5,1)
    plt.imshow(minimap_height_map)
    plt.subplot(5,5,2)
    plt.imshow(minimap_visibility_map)
    plt.subplot(5,5,3)
    plt.imshow(minimap_creep)
    plt.subplot(5,5,4)
    plt.imshow(minimap_camera)
    plt.subplot(5,5,5)
    plt.imshow(minimap_player_id)
    plt.subplot(5,5,6)
    plt.imshow(minimap_player_relative)
    plt.subplot(5,5,7)
    plt.imshow(minimap_selected)
    plt.subplot(5,5,8)
    plt.imshow(screen_height_map)
    plt.subplot(5,5,9)
    plt.imshow(screen_visibility_map)
    plt.subplot(5,5,10)
    plt.imshow(screen_creep)
    plt.subplot(5,5,11)
    plt.imshow(screen_power)
    plt.subplot(5,5,12)
    plt.imshow(screen_player_id)
    plt.subplot(5,5,13)
    plt.imshow(screen_player_relative)
    plt.subplot(5,5,14)
    plt.imshow(screen_unit_type)
    plt.subplot(5,5,15)
    plt.imshow(screen_selected)
    plt.subplot(5,5,16)
    plt.imshow(screen_unit_hit_point)
    plt.subplot(5,5,17)
    plt.imshow(screen_unit_hit_points_ratio)
    plt.subplot(5,5,18)
    plt.imshow(screen_unit_energy)
    plt.subplot(5,5,19)
    plt.imshow(screen_unit_energy_ratio)
    plt.subplot(5,5,20)
    plt.imshow(screen_unit_shield)
    plt.subplot(5,5,21)
    plt.imshow(screen_unit_shield_ratio)
    plt.subplot(5,5,22)
    plt.imshow(screen_unit_density)
    plt.subplot(5,5,23)
    plt.imshow(screen_unit_density_aa)
    plt.subplot(5,5,24)
    plt.imshow(screen_effects)
    plt.show()

    action = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
    obs = env.step(actions=[action])
    reward = obs[0].reward
    done = obs[0].step_type == environment.StepType.LAST

    if done:
        obs = env.reset()