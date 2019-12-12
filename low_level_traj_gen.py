import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import daisy_kinematics

TRIPOD_LEG_PAIR_1 = [0, 3, 4]
TRIPOD_LEG_PAIR_2 = [1, 2, 5]
NUM_LEGS = 6
EPSILON = 1e-4

def get_foot_position_world_from_com(state):
    '''
    Forwark kinematic for getting the foot position in com then translate
    Input:
            state: a dict contains multiple information of robot
    return:
            foot_position_world: [6*3] foot position in the world frame(attached to the CoM)
    '''
    foot_pos_com = daisy_kinematics.FK_CoM2Foot(state['j_pos'])
    return daisy_kinematics.World2Foot(foot_pos_com,state['base_ori_euler'])

class IK_traj_generator():
    def __init__(self,
        init_state,
        leg_clearance = 0.15,
            ):
        self.leg_clearance = leg_clearance
        self.swing_set = TRIPOD_LEG_PAIR_1 
        self.stance_set = TRIPOD_LEG_PAIR_2
        self.init_foot_pos =  daisy_kinematics.FK_CoM2Foot(init_state['j_pos'])
        self.standing_height = - self.init_foot_pos[0][2]
        self.des_body_ori = np.zeros(3)
        
    def update_latent_action(self,state,latent_action):
        # update latent action
        self.latent_action = latent_action
        # update swing/stance leg set
        self.swing_set, self.stance_set = np.copy(self.stance_set), np.copy(self.swing_set)
        self.swing_start_foot_pos = daisy_kinematics.FK_CoM2Foot(state['j_pos'])

    def get_action(self, state, phase):
        des_foot_pos = []
        des_foot_height = (self.leg_clearance * math.sin(math.pi * phase + EPSILON) - self.standing_height)
        for i in range(NUM_LEGS):
            des_single_foot_pos = np.zeros(3)
            des_single_foot_pos[:2] = ((self.latent_action[:2] -self.swing_start_foot_pos[i][:2]+ self.init_foot_pos[i][:2] ) * phase 
                                            + self.swing_start_foot_pos[i][:2])
            if i in self.swing_set:
                des_single_foot_pos[2] = des_foot_height
            else:
                des_single_foot_pos[2] = - self.standing_height 
            
            des_foot_pos.append(des_single_foot_pos)
        self.des_foot_position_world = np.array(des_foot_pos)
        des_foot_position_com = daisy_kinematics.Foot2World(self.des_foot_position_world,self.des_body_ori)
        des_leg_pose = daisy_kinematics.IK_foot2CoM(des_foot_position_com)
        return  des_leg_pose 
    
# class NN_traj_generator():
#     def __init__(self):

#     def get_action(self):
    
#     def update_TG(self):
#         # try ARS???

class low_level_TG():
    def __init__(self, 
        z_dim,
        a_dim,
        num_timestep_per_footstep,
        batch_size,
        low_level_policy_type,
        update_low_level_policy,
        update_low_level_policy_lr,
        init_state,
        **kwargs
    ):
        # Initialize trajectory generator
        self.z_dim = z_dim
        self.a_dim = a_dim
        self.num_timestep_per_footstep = num_timestep_per_footstep

        if low_level_policy_type == 'IK':
            self.policy = IK_traj_generator(init_state)

        self.update_low_level_policy = update_low_level_policy
        if self.update_low_level_policy:
            self.batch_size = batch_size
            self.policy_lr = update_low_level_policy_lr
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(),lr=self.policy_lr)

    def update_latent_action(self,state, latent_action):
        self.policy.update_latent_action(state, latent_action)

    def get_action(self,state, t):
        phase = float(t)/self.num_timestep_per_footstep
        action = self.policy.get_action(state, phase)
        return action 

    def update_TG(self):
        if self.update_low_level_policy:
            self.policy.update()