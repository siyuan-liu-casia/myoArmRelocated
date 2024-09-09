import gym
import myosuite
import numpy as np
from gym.envs.registration import register

register(id='Relocate-v1',
        entry_point='relocate_v1:RelocateEnvV1',
        max_episode_steps=500,
        kwargs={
            #路径更换设备需要重设置一下
            'model_path': r'C:\Users\user\anaconda3\envs\myo\Lib\site-packages\myosuite\envs\myo\assets\arm\myoarm_relocate_v1.xml',
            'normalize_act': True,
            'frame_skip': 5,
            'pos_th': 0.1,              # cover entire base of the receptacle
            'rot_th': np.inf,           # ignore rotation errors
            'target_xyz_range': {'high':[0.1, -.35, 1.2], 'low':[0.1, -.35, 1.2]},
            'target_rxryrz_range': {'high':[0.0, 0.0, 0.0], 'low':[0.0, 0.0, 0.0]}
        }
    )

env = gym.make('Relocate-v1')
env.reset()

# geom_0_indices = np.where(env.sim.model.geom_group == 0)
# env.sim.model.geom_rgba[geom_0_indices[0][0], 3] = 0 # to make the floor transparent
# env.sim.model.geom_rgba[geom_0_indices[0][1], 3] = 0 # to make the room transparent

# env.sim.model.geom_rgba[geom_0_indices[0][2], 3] = 0 # to make the body transparent
# env.sim.model.geom_rgba[geom_0_indices[0][3], 3] = 0 # to make the large arm transparent

geom_1_indices = np.where(env.sim.model.geom_group == 1)
env.sim.model.geom_rgba[geom_1_indices[0][:], 3] = 0 # to make the skin transparent

site_2_indices = np.where(env.sim.model.site_group == 2)
env.sim.model.site_rgba[site_2_indices[0][:], 3] = 0 # to make the site transparent

for _ in range(1000):
    env.mj_render()
    env.step(env.action_space.sample()) # take a random action
env.close()