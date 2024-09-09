import gym
import myosuite 
from stable_baselines3 import PPO
import os
import time
from gym.envs.registration import register
import numpy as np
from matplotlib import pyplot as plt
import datetime
import torch as th
import skvideo.io
from plot import *

current_dir = os.path.dirname(os.path.abspath(__file__))
# Register MyoSuite Envs
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

env_test = gym.make('Relocate-v1')

nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

#输入你需要测试的model的位置
model_dir = r'G:\pycode\MyoSuite\PPO_train\models\PPO-2048-64-10-Time-2024-04-14-11-16\PPO 5400000_steps.zip'
train_test_dir = os.path.join(current_dir, f"Test/Time-{nowtime}")
os.makedirs(train_test_dir, exist_ok=True)

pi = PPO.load(model_dir, env=env_test)

frames = []
done = False
state = env_test.reset()

done = False
MuscleExcitation = []
for _ in range(500):
    frame = env_test.sim.renderer.render_offscreen(
                    width=400,
                    height=400,
                    camera_id=2)
    frames.append(frame)
    o = env_test.get_obs()
    a = pi.predict(o, deterministic=True)[0]
    MuscleExcitation.append(a)
    next_o, r, done, ifo = env_test.step(a) # take an action based on the current observation
    if done:
        break 
skvideo.io.vwrite(f'{train_test_dir}/Relocte.mp4', np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})
plot_and_save_muscle_excitation(MuscleExcitation, train_test_dir)
env_test.close()
