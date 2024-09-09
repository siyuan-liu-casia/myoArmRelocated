# MyoArm relocate_v1 env

Move the box into the red target ball located in front of the chest.



https://github.com/user-attachments/assets/7b2c7046-9f85-4a79-bbf0-105da43b5d1e



![环境](https://picgo-liusiyuan.oss-cn-beijing.aliyuncs.com/picgo-lsy/202409081807645.png)

## Environment Setup

(Since this task is not included in myosuite, you need to configure the simulation files for the task in your conda env.)

1.Copy the arm folder to C:\Users\user\anaconda3\envs\myo\Lib\site-packages\myosuite\envs\myo\assets, which is located in the assets directory of the myosuite package in your conda environment.

2.Copy the assets folder to C:\Users\user\anaconda3\envs\myo\Lib\site-packages\myosuite\simhive\myo_sim\arm, which is in the myo_sim\arm directory of the myosuite package in your conda environment.

3.In the register function, set the model_path to the XML file path from step 1: C:\Users\user\anaconda3\envs\myo\Lib\site-packages\myosuite\envs\myo\assets\arm\myoarm_relocate_v1.xml.

## **Environment Testing**

Run envtest.py to check if the Relocate-v1 environment is functioning. The arm should move randomly with varying muscle activations.

## **register Parameter Description**

max_episode_steps=500: Maximum number of steps per episode.

normalize_act: True, # Normalize muscle activation values between -1 and 1.

pos_th: 0.1, # Distance threshold for the target. The task is considered successful when the center of the object is within this distance from the target.

rot_th: np.inf, # Rotation threshold for the target. This specifies how much the object’s orientation can deviate from the target orientation. Here, it is set to infinity, meaning orientation does not matter for success.

target_xyz_range: {'high': [0.1, -.35, 1.2], 'low': [0.1, -.35, 1.2]}, # The target position is randomly generated for each episode. If the high and low values are the same, the target will be fixed; otherwise, it will be generated randomly within this range.

target_rxryrz_range: {'high': [0.0, 0.0, 0.0], 'low': [0.0, 0.0, 0.0]} # The range for random target orientations.

## Files

relocate_v1.py: Environment code.

PPO_train.py: Runs PPO training in a loop. Every 200,000 timesteps, a test is performed, followed by continued training. A total of 50 runs, each with 200,000 steps, sums up to 10 million timesteps.

PPO_train_v2.py: Trains PPO for 10 million steps directly and then runs the test.

test.py: Tests a trained model.

plot.py: Plots the muscle activation curves.
