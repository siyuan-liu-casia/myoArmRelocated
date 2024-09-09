""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates. Institute of automation, Chinese Academy of Science
Authors: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Liu siyuan(liusiyuan2023@ia.ac.cn)
2024.4.13
================================================= """
import collections
import numpy as np
import gym

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import mat2euler, euler2quat

class RelocateEnvV1(BaseV0):
    #observation 键:关节位置、关节角度、物体位置、目标位置、位置误差、物体旋转角度、目标旋转角度、旋转误差
    #get_obs_dict()获得的obs字典，只有在DEFAULT_OBS_KEYS的会作为实际的obs，其余量可测但没有使用? 这里可能不太对 理解有问题
    DEFAULT_OBS_KEYS = ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot', 'rot_err']
    #奖励权重 可根据rwd_dict中的内容，添加相关项并调整系数
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_dist": 50,
        "rot_dist": 0,
        "reach_dist": 100,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # Two step construction (init+setup) is required for pickling to work correctly.
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)


    def _setup(self,
            target_xyz_range,        # 目标位置范围（相对于初始位置）
            target_rxryrz_range,     # 目标旋转范围（相对于初始旋转）
            obj_xyz_range = None,    # 物体位置范围（相对于初始位置）
            obj_geom_range = None,   # 物体几何范围的随机化大小
            obj_mass_range = None,   # 物体大小范围
            obj_friction_range = None,# 物体摩擦范围
            qpos_noise_range = None, # 初始化时关节空间中的噪声
            obs_keys:list = DEFAULT_OBS_KEYS,  # 观测键列表
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,  # 奖励键和权重列表
            pos_th = .025,          # 位置误差阈值
            rot_th = 0.262,         # 旋转误差阈值
            drop_th = 0.50,         # 掉落高度阈值
            **kwargs,
        ):
        self.palm_sid = self.sim.model.site_name2id("S_grasp")  # 手掌site的ID
        self.object_sid = self.sim.model.site_name2id("object_o")  # 物体site的ID
        self.object_bid = self.sim.model.body_name2id("Object")  # 物体body的ID
        self.goal_sid = self.sim.model.site_name2id("target_o")  # 目标site的ID
        self.success_indicator_sid = self.sim.model.site_name2id("target_ball")  # 成功指示器site的ID
        self.goal_bid = self.sim.model.body_name2id("target")  # 目标body的ID
        self.target_xyz_range = target_xyz_range  # 目标位置范围
        self.target_rxryrz_range = target_rxryrz_range  # 目标旋转范围
        self.obj_geom_range = obj_geom_range  # 物体几何范围
        self.obj_mass_range = obj_mass_range  # 物体质量范围
        self.obj_friction_range = obj_friction_range  # 物体摩擦范围
        self.obj_xyz_range = obj_xyz_range  # 物体位置范围
        self.qpos_noise_range = qpos_noise_range  # 关节空间中的噪声范围
        self.pos_th = pos_th  # 位置误差阈值
        self.rot_th = rot_th  # 旋转误差阈值
        self.drop_th = drop_th  # 丢弃高度阈值

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    **kwargs,
        )
        keyFrame_id = 0 if self.obj_xyz_range is None else 1
        self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy()

    #获取 observation 字典
    def get_obs_dict(self, sim):      
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])                #当前仿真的时间
        obs_dict['hand_qpos'] = sim.data.qpos[:-7].copy()           #手部的关节位置
        obs_dict['hand_qpos_corrected'] = sim.data.qpos[:-6].copy() #修正后的手部关节位置 见https://github.com/MyoHub/myosuite/issues/98
        obs_dict['hand_qvel'] = sim.data.qvel[:-6].copy()*self.dt   #手部的关节速度
        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]   #物体的位置
        obs_dict['goal_pos'] = sim.data.site_xpos[self.goal_sid]    #目标位置
        obs_dict['palm_pos'] = sim.data.site_xpos[self.palm_sid]    #手掌的位置
        obs_dict['pos_err'] = obs_dict['goal_pos'] - obs_dict['obj_pos']    #物体到目标的位置误差
        obs_dict['reach_err'] = obs_dict['palm_pos'] - obs_dict['obj_pos']  #手掌到物体的距离
        obs_dict['obj_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.object_sid],(3,3)))  #物体的旋转
        obs_dict['goal_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.goal_sid],(3,3)))   #目标的旋转
        obs_dict['rot_err'] = obs_dict['goal_rot'] - obs_dict['obj_rot']    #物体到目标的旋转误差

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()    #上一时间步的肌肉激活
        return obs_dict

    #获得奖励字典
    def get_reward_dict(self, obs_dict):            
        reach_dist = np.abs(np.linalg.norm(self.obs_dict['reach_err'], axis=-1))    #手掌到物体的距离
        pos_dist = np.abs(np.linalg.norm(self.obs_dict['pos_err'], axis=-1))        #物体到目标的位置误差
        rot_dist = np.abs(np.linalg.norm(self.obs_dict['rot_err'], axis=-1))        #物体到目标的旋转误差
        # sim.model.na 肌肉数量
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0   #归一化的肌肉激活 
        drop = reach_dist > self.drop_th
        rwd_dict = collections.OrderedDict((
            # 执行奖励调整 --
            # 更新下面的可选键部分
            # 根据需要更新奖励键（DEFAULT_RWD_KEYS_AND_WEIGHTS）以更新最终奖励
            # 例如：环境预先提供了两个键pos_dist和reach_dist

            ('pos_dist', -1.*pos_dist),  # 位置距离的负值
            ('rot_dist', -1.*rot_dist),  # 旋转距离的负值
            ('reach_dist', -1.*reach_dist),  # 到达距离的负值
            ('act_reg', -1.*act_mag),  # 肌肉激活的负值
            ('sparse', -rot_dist-10.0*pos_dist),  # 稀疏奖励
            ('solved', (pos_dist<self.pos_th) and (rot_dist<self.rot_th) and (not drop)),  # 是否解决问题
            ('done', drop),  # 物体是否掉落
        ))
        #奖励 为rwd_dict和rwd_keys_wt(即DEFAULT_RWD_KEYS_AND_WEIGHTS)中对应项相乘的结果
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)

        # Success Indicator 如果方块在范围内，即solved，将显示的目标球设为绿色并放大
        self.sim.model.site_rgba[self.success_indicator_sid, :2] = np.array([0, 2]) if rwd_dict['solved'] else np.array([2, 0])
        self.sim.model.site_size[self.success_indicator_sid, :] = np.array([.25,]) if rwd_dict['solved'] else np.array([0.1,])
        return rwd_dict

    #我也不知道是啥
    def get_metrics(self, paths, successful_steps=5):
        """
        Evaluate paths and report metrics
        """
        num_success = 0
        num_paths = len(paths)

        # average sucess over entire env horizon
        for path in paths:
            # record success if solved for provided successful_steps
            if np.sum(path['env_infos']['rwd_dict']['solved'] * 1.0) > successful_steps:
                num_success += 1
        score = num_success/num_paths

        # average activations over entire trajectory (can be shorter than horizon, if done) realized
        effort = -1.0*np.mean([np.mean(p['env_infos']['rwd_dict']['act_reg']) for p in paths])

        metrics = {
            'score': score,
            'effort':effort,
            }
        return metrics

    
    def reset(self, reset_qpos=None, reset_qvel=None):
        # 随机设置目标位置和旋转
        self.sim.model.body_pos[self.goal_bid] = self.np_random.uniform(**self.target_xyz_range)  # 目标位置
        self.sim.model.body_quat[self.goal_bid] = euler2quat(self.np_random.uniform(**self.target_rxryrz_range))  # 目标旋转

        # 随机设置物体位置
        if self.obj_xyz_range is not None:
            self.sim.model.body_pos[self.object_bid] = self.np_random.uniform(**self.obj_xyz_range)

        # 随机设置物体几何形状和位置
        if self.obj_geom_range is not None:
            for body in ["Object", ]:
                bid = self.sim.model.body_name2id(body)
                for gid in range(self.sim.model.body_geomnum[bid]):
                    gid += self.sim.model.body_geomadr[bid]  # 获取几何体ID
                    # 更新类型、大小和碰撞边界
                    self.sim.model.geom_type[gid] = self.np_random.randint(low=2, high=7)  # 随机形状
                    self.sim.model.geom_size[gid] = self.np_random.uniform(low=self.obj_geom_range['low'], high=self.obj_geom_range['high'])  # 随机大小
                    self.sim.model.geom_aabb[gid][3:] = self.obj_geom_range['high']  # 边界框，(中心，大小)
                    self.sim.model.geom_rbound[gid] = 2.0 * max(self.obj_geom_range['high'])  # 边界球的半径

                    self.sim.model.geom_pos[gid] = self.np_random.uniform(low=-1.0 * self.sim.model.geom_size[gid], high=self.sim.model.geom_size[gid])  # 随机位置
                    self.sim.model.geom_quat[gid] = euler2quat(self.np_random.uniform(low=(-np.pi/2, -np.pi/2, -np.pi/2), high=(np.pi/2, np.pi/2, np.pi/2)))  # 随机旋转
                    self.sim.model.geom_rgba[gid] = self.np_random.uniform(low=[.2, .2, .2, 1], high=[.9, .9, .9, 1])  # 随机颜色

                    # 摩擦变化
                    if self.obj_friction_range is not None:
                        self.sim.model.geom_friction[gid] = self.np_random.uniform(**self.obj_friction_range)

                # 质量变化
                if self.obj_mass_range is not None:
                    self.sim.model.body_mass[self.object_bid] = self.np_random.uniform(**self.obj_mass_range)

                self.sim.forward()

        # 初始手臂姿势
        if self.qpos_noise_range is not None:
            reset_qpos_local = self.init_qpos + self.qpos_noise_range * (self.sim.model.jnt_range[:, 1] - self.sim.model.jnt_range[:, 0])
            reset_qpos_local[-6:] = self.init_qpos[-6:]
        else:
            reset_qpos_local = reset_qpos

        obs = super().reset(reset_qpos_local, reset_qvel)
        if self.sim.data.ncon > 0:
            self.reset(reset_qpos, reset_qvel)

        return obs
