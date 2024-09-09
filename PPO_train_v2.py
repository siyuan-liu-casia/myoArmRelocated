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
import os
import skvideo.io
from plot import *
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# Register MyoSuite Envs
register(id='Relocate-v1',
        entry_point='relocate_v1:RelocateEnvV1',
        max_episode_steps=500,
        kwargs={
            'model_path': r'C:\Users\user\anaconda3\envs\myo\Lib\site-packages\myosuite\envs\myo\assets\arm\myoarm_relocate_v1.xml',
            'normalize_act': True,
            'frame_skip': 5,
            'pos_th': 0.1,              # cover entire base of the receptacle
            'rot_th': np.inf,           # ignore rotation errors
            'target_xyz_range': {'high':[0.1, -.35, 1.2], 'low':[0.1, -.35, 1.2]},
            'target_rxryrz_range': {'high':[0.0, 0.0, 0.0], 'low':[0.0, 0.0, 0.0]}
        }
    )


seed = 123

env = gym.make('Relocate-v1')
env.reset()
env.seed(seed)
env.sim.renderer.set_free_camera_settings()


nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

model_dir = os.path.join(current_dir, f"models/PPO-seed-{seed}-Time-{nowtime}")
loggir_dir = os.path.join(current_dir, f"logs/PPO-seed-{seed}-Time-{nowtime}")
train_test_dir = os.path.join(current_dir, f"train_test/PPO-seed-{seed}-Time-{nowtime}")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(loggir_dir, exist_ok=True)
os.makedirs(train_test_dir, exist_ok=True)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=loggir_dir, seed=seed)

total_timesteps = 10_000_000

model.learn(total_timesteps=total_timesteps)
model.save(model_dir)

pi = PPO.load(model_dir, env=env)
frames = []
done = False

state = env.reset()
done = False
MuscleExcitation = []
for _ in range(500):
    frame = env.sim.renderer.render_offscreen(
                    width=400,
                    height=400,
                    camera_id=2)
    frames.append(frame)
    o = env.get_obs()
    a = pi.predict(o, deterministic=True)[0]
    MuscleExcitation.append(a)
    next_o, r, done, ifo = env.step(a) # take an action based on the current observation
    if done:
        break 
skvideo.io.vwrite(f'{train_test_dir}/Relocte.mp4', np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})
plot_and_save_muscle_excitation(MuscleExcitation, train_test_dir)
env.close()
