import gym
import myosuite 
from stable_baselines3 import PPO
import os
import time
from gym.envs.registration import register
# from gymnasium.envs.registration import register
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

seed = 123

env = gym.make('Relocate-v1')
env.seed(seed)
env.reset()
env.sim.renderer.set_free_camera_settings()

#当前时间
nowtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
#模型存储路径、tensorboard存储路径、训练测试结果路径
model_dir = os.path.join(current_dir, f"models/PPO-seed-{seed}-Time-{nowtime}")
loggir_dir = os.path.join(current_dir, f"logs/PPO-seed-{seed}-Time-{nowtime}")
train_test_dir = os.path.join(current_dir, f"train_test/PPO-seed-{seed}-Time-{nowtime}")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(loggir_dir, exist_ok=True)
os.makedirs(train_test_dir, exist_ok=True)


# 收敛最好
# policy_kwargs = dict(activation_fn=th.nn.Tanh,
#                         net_arch=dict(pi=[256, 512, 256], 
#                         vf=[256, 512, 256]),
#                         share_features_extractor=True)

#ppo算法
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=loggir_dir, seed=seed)

#这个代码是, 每200000个timestep之后，测试一次，然后继续训练，共训练200000*50 = 10M 步
TimeSteps = 200000

# 循环训练模型
for i in range(1, 50):
    model.learn(total_timesteps=TimeSteps, reset_num_timesteps=False)
    save_dir = model_dir + "/PPO " + str(i * TimeSteps) + "_steps"
    model.save(save_dir)    

    plotdir = f"{train_test_dir}/time_step_{i * TimeSteps}"
    os.makedirs(plotdir, exist_ok=True)

    # 加载训练好的模型
    pi = PPO.load(save_dir, env=env)
    frames = []
    done = False

    state = env.reset()
    done = False
    MuscleExcitation = []

    # 运行环境测试 录视频 画肌肉激活
    for _ in range(500):
        frame = env.sim.renderer.render_offscreen(
            width=400,
            height=400,
            camera_id=2)
        frames.append(frame)
        o = env.get_obs()
        a = pi.predict(o, deterministic=True)[0]
        MuscleExcitation.append(a)
        next_o, r, done, info = env.step(a)  # 根据当前观察结果采取行动
        if done:
            break

    # 保存视频和肌肉激活图
    skvideo.io.vwrite(f'{plotdir}/Relocte.mp4', np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})
    plot_and_save_muscle_excitation(MuscleExcitation, plotdir)

env.close()