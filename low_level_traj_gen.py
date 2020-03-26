import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from daisy_raibert_controller import get_foot_position_world_from_com
import daisy_kinematics

import utils

# Tripod
TRIPOD_LEG_PAIR_1 = [0, 3, 4]
TRIPOD_LEG_PAIR_2 = [1, 2, 5]
# # right tri
# TRIPOD_LEG_PAIR_2 = [0, 5]
# TRIPOD_LEG_PAIR_1 = [1, 2, 3, 4]
# TRIPOD_LEG_PAIR_1 = [0, 1,3]
# TRIPOD_LEG_PAIR_2 = [2, 4, 5]
# # left tri
# TRIPOD_LEG_PAIR_1 = [0, 1 ,2]
# TRIPOD_LEG_PAIR_2 = [3, 4, 5]
# TRIPOD_LEG_PAIR_2 = [0, 1 ,2]
# TRIPOD_LEG_PAIR_1 = [3, 4, 5]
# # tri small big
# TRIPOD_LEG_PAIR_2 = [0, 2 ,3]
# TRIPOD_LEG_PAIR_1 = [1, 4 ,5]
# inv tri small big
# TRIPOD_LEG_PAIR_1 = [0, 1 ,5]
# TRIPOD_LEG_PAIR_2 = [2, 3 ,4]
# # tri big small
# TRIPOD_LEG_PAIR_1 = [0, 4 ,5]
# TRIPOD_LEG_PAIR_2 = [1, 2, 3]

# # inv tri big small
# TRIPOD_LEG_PAIR_1 = [0, 1 ,4]
# TRIPOD_LEG_PAIR_2 = [2, 3 ,5]


NUM_LEGS = 6
EPSILON = 1e-4


class IK_traj_generator():
    def __init__(self,
        init_state,
        leg_clearance = 0.2,
            ):
        self.leg_clearance = leg_clearance
        self.swing_set = TRIPOD_LEG_PAIR_1 
        self.stance_set = TRIPOD_LEG_PAIR_2
        self.init_foot_pos =  get_foot_position_world_from_com(init_state)
        self.standing_height = - self.init_foot_pos[0][2]
        self.last_des_body_ori = np.zeros(3)
        self.des_body_ori = np.zeros(3)

        self.init_r_yaw = utils.get_init_r_yaw(self.init_foot_pos)
        
    def update_swing_stance(self):
        self.swing_set, self.stance_set = np.copy(self.stance_set), np.copy(self.swing_set)

    
    def update_latent_action_params(self,state,latent_action):
        # update latent action
        self.latent_action = latent_action
        # update swing/stance leg set
        self.swing_start_foot_pos = get_foot_position_world_from_com(state)
        self.last_com_ori = state['base_ori_euler']
        self.last_des_body_ori[2] = self.last_des_body_ori[2] + self.latent_action[-1] # TODO: pretty annoying, need to change

        self.target_delta_xy = np.zeros((NUM_LEGS, 2))
        for i in range(NUM_LEGS):
            if i in self.swing_set:
                angle = self.last_des_body_ori[2]  + self.init_r_yaw[i][1]
                self.target_delta_xy[i][0] = self.init_r_yaw[i][0] * math.cos(angle) + self.latent_action[0] - self.swing_start_foot_pos[i][0]                              
                self.target_delta_xy[i][1] = self.init_r_yaw[i][0] * math.sin(angle) + self.latent_action[1] - self.swing_start_foot_pos[i][1]                                                            
            else:
                angle = self.last_com_ori[2] + self.init_r_yaw[i][1]
                self.target_delta_xy[i][0] = self.init_r_yaw[i][0] * math.cos(angle) - self.latent_action[0] - self.swing_start_foot_pos[i][0]
                self.target_delta_xy[i][1] = self.init_r_yaw[i][0] * math.sin(angle) - self.latent_action[1] - self.swing_start_foot_pos[i][1]


    def update_latent_action(self,state,latent_action):
        self.update_swing_stance()
        self.update_latent_action_params(state,latent_action)
        
        
    def get_action(self, state, phase):
        des_foot_pos = []
        self.des_body_ori[2] = (self.last_des_body_ori[2] - self.last_com_ori[2]) * phase + self.last_com_ori[2]
        des_foot_height = (self.leg_clearance * math.sin(math.pi * phase + EPSILON) - self.standing_height)

        for i in range(NUM_LEGS):
            des_single_foot_pos = np.zeros(3)
            if i in self.swing_set:
                des_single_foot_pos[2] = des_foot_height                                                      
            else:
                des_single_foot_pos[2] = - self.standing_height 

            des_single_foot_pos[0] =  self.target_delta_xy[i][0] * phase + self.swing_start_foot_pos[i][0]
            des_single_foot_pos[1] =  self.target_delta_xy[i][1] * phase + self.swing_start_foot_pos[i][1] 
            des_foot_pos.append(des_single_foot_pos)

        self.des_foot_position_world = np.array(des_foot_pos)             
        self.des_foot_position_com = daisy_kinematics.Foot2World(self.des_foot_position_world,self.des_body_ori)
        self.des_foot_position_com_swing = daisy_kinematics.Foot2World(self.des_foot_position_world,np.array(state['base_ori_euler']))
        # for i in self.swing_set:
        #     self.des_foot_position_com[i] = self.des_foot_position_com_swing[i]
        des_leg_pose = daisy_kinematics.IK_foot2CoM(self.des_foot_position_com)
        
        return  des_leg_pose 



###########################################################################################################
class IK_CoM_traj_generator():
    def __init__(self,
        init_state,
        leg_clearance = 0.2,
            ):
        self.leg_clearance = leg_clearance
        self.swing_set = TRIPOD_LEG_PAIR_1 
        self.stance_set = TRIPOD_LEG_PAIR_2
        self.init_foot_pos =  get_foot_position_world_from_com(init_state)
        self.standing_height = - self.init_foot_pos[0][2]
        self.last_des_body_ori = np.array([0,0,init_state['base_ori_euler'][2]])
        self.des_body_ori = np.array([0,0,init_state['base_ori_euler'][2]])
        self.swing_stance_idx = 0
        self.init_r_yaw = utils.get_init_r_yaw(self.init_foot_pos)

    def update_swing_stance(self):
        self.swing_set, self.stance_set = np.copy(self.stance_set), np.copy(self.swing_set)
        # self.swing_set = [self.swing_stance_idx%6]
        # self.swing_stance_idx += 1 

    def update_latent_action_params(self,state,latent_action):
        # transform latent action to world frame 
         # update latent action
        self.latent_action = latent_action
        # update
        self.swing_start_foot_pos = daisy_kinematics.FK_CoM2Foot(state['j_pos'])
        self.swing_start_foot_pos_world = get_foot_position_world_from_com(state)

        self.last_com_ori = state['base_ori_euler']
        self.last_des_body_ori[2] = self.last_com_ori[2]  + self.latent_action[-1] # TODO: pretty annoying, need to change

        self.target_delta_xy = np.zeros((NUM_LEGS, 3))
        for i in range(NUM_LEGS):
            if i in self.swing_set:
                angle = self.latent_action[-1]  + self.init_r_yaw[i][1]
                self.target_delta_xy[i][0] = self.init_r_yaw[i][0] * math.cos(angle) + self.latent_action[0] - self.swing_start_foot_pos[i][0]                              
                self.target_delta_xy[i][1] = self.init_r_yaw[i][0] * math.sin(angle) + self.latent_action[1] - self.swing_start_foot_pos[i][1]                                                            
            else:
                angle =  self.init_r_yaw[i][1]
                self.target_delta_xy[i][0] = self.init_r_yaw[i][0] * math.cos(angle) - self.latent_action[0] - self.swing_start_foot_pos[i][0]
                self.target_delta_xy[i][1] = self.init_r_yaw[i][0] * math.sin(angle) - self.latent_action[1] - self.swing_start_foot_pos[i][1]
            self.target_delta_xy[i][2] = -self.standing_height
        # transform target x y to the 0 yaw frame
        self.target_delta_xyz_world = daisy_kinematics.World2Foot(self.target_delta_xy, np.array(self.last_com_ori))


    def update_latent_action(self,state,latent_action):
        self.update_swing_stance()
        self.update_latent_action_params(state,latent_action)

    def get_action(self, state, phase):
        des_foot_pos = []
        self.des_body_ori[2] = (self.last_des_body_ori[2] - self.last_com_ori[2]) * phase + self.last_com_ori[2]
        # this seems to be designed only when walking on a flat ground 
        des_foot_height_delta = (self.leg_clearance * math.sin(math.pi * phase + EPSILON))

        for i in range(NUM_LEGS): 
            des_single_foot_pos = np.zeros(3)
            if i in self.swing_set:
                des_single_foot_pos[2] = des_foot_height_delta - self.standing_height                                        
            else:
                des_single_foot_pos[2] = - 0.0*des_foot_height_delta - self.standing_height 

            des_single_foot_pos[0] =  self.target_delta_xyz_world[i][0] * phase + self.swing_start_foot_pos_world[i][0]
            des_single_foot_pos[1] =  self.target_delta_xyz_world[i][1] * phase + self.swing_start_foot_pos_world[i][1] 
            des_foot_pos.append(des_single_foot_pos)

        self.des_foot_position_world = np.array(des_foot_pos)             
        self.des_foot_position_com = daisy_kinematics.Foot2World(self.des_foot_position_world,self.des_body_ori)

        des_leg_pose = daisy_kinematics.IK_foot2CoM(self.des_foot_position_com)
        
        return  des_leg_pose 



#################################################################################################################################
class NN_tra_generator(nn.Module):
    '''
    The NN trajectory generator(low level controller) is a NN which is trained in a supervised manner
    '''
    def __init__(self, z_dim, policy_output_dim, policy_hidden_num, device) :
        '''
        Initialize the structure of trajectory generator
        '''
        super().__init__()

        self.device = device
        self.trunk = nn.Sequential(
            nn.Linear(1 + z_dim + 0, policy_hidden_num ), nn.ReLU(),
            nn.Linear(policy_hidden_num, policy_hidden_num), nn.ReLU(),
            nn.Linear(policy_hidden_num, policy_output_dim))
        self.z_action = np.zeros(z_dim)
        self.action_mean_std = torch.FloatTensor(np.load('./LL_mean_std.npy')).to(self.device)

    def forward(self, z_action, state_vec):
        low_level_input = torch.cat([z_action, state_vec], dim=1)
        return utils.inverse_normalization(self.trunk(low_level_input), self.action_mean_std[0], self.action_mean_std[1])
        # return self.trunk(low_level_input)

    def update_latent_action(self, state, latent_action):
        self.z_action = latent_action

    def get_action(self,state,  phase):
        # j_state = state['j_pos']
        j_state = np.array([])
        state_vec = np.append(j_state,[phase])
        latent_action = torch.FloatTensor(self.z_action).to(self.device)
        latent_action = latent_action.unsqueeze(0)
        state_term = torch.FloatTensor(state_vec).to(self.device)
        state_term = state_term.unsqueeze(0)
        action = self.forward(latent_action, state_term)
        return action.cpu().data.numpy().flatten() 

class decentralized_NN_policy(nn.Module):
    def __init__(self, z_dim, policy_output_dim, policy_hidden_num, device) :
        '''
        Initialize the structure of trajectory generator
        '''
        super().__init__()

        self.device = device
        self.trunk = nn.Sequential(
            nn.Linear(1 + z_dim, policy_hidden_num ), nn.ReLU(),
            nn.Linear(policy_hidden_num, policy_hidden_num), nn.ReLU(),
            nn.Linear(policy_hidden_num, policy_output_dim))
        self.z_action = np.zeros(z_dim)

    def forward(self, z_action, phase):
        low_level_input = torch.cat([z_action, phase], dim=1)
        return self.trunk(low_level_input)

    def update_latent_action(self, state, latent_action):
        self.z_action = latent_action.reshape((6,1))

    def get_action(self, state, phase):
        phase_vec = phase * np.ones(6,1)
        z_action_tensor = torch.as_tensor(self.z_action, device = self.device).float()
        phase_tensor = torch.as_tensor(phase_vec, device = self.device).float()
        action_tensor = self.forward(z_action_tensor, phase_tensor)
        return action_tensor.cpu().data.numpy().reshape((18))


class low_level_TG():
    def __init__(self, 
        device,
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
        self.device = device
        self.z_dim = z_dim
        self.a_dim = a_dim
        self.num_timestep_per_footstep = num_timestep_per_footstep
        self.low_level_policy_type = low_level_policy_type
        self.init_state = init_state

        self.action_limit = np.empty((18,2))
        for p in range(6):
            self.action_limit[3*p][0] = self.init_state['j_pos'][3*p]+0.6
            self.action_limit[3*p][1] = self.init_state['j_pos'][3*p]-0.6
            self.action_limit[3*p+1][0] = self.init_state['j_pos'][3*p+1]+0.4
            self.action_limit[3*p+1][1] = self.init_state['j_pos'][3*p+1]-0.4
            self.action_limit[3*p+2][0] = self.init_state['j_pos'][3*p+2]+0.4
            self.action_limit[3*p+2][1] = self.init_state['j_pos'][3*p+2]-0.4

        if low_level_policy_type == 'IK':
            self.policy = IK_CoM_traj_generator(init_state)
        elif low_level_policy_type =='NN':
            self.policy = NN_tra_generator( z_dim = z_dim, policy_output_dim = a_dim, policy_hidden_num = 512, device = device)
        
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

        for p in range(6):
            action[3*p] = np.clip(action[3*p],self.action_limit[3*p][1],self.action_limit[3*p][0])
            # action[3*p+1] = np.clip(action[3*p+1],self.action_limit[3*p+1][1],self.action_limit[3*p+1][0])
            action[3*p+2] = np.clip(action[3*p+2],self.action_limit[3*p+2][1],self.action_limit[3*p+2][0])
        return action 

    def update_TG(self):
        if self.update_low_level_policy:
            self.policy.update()

    def reset(self,state):
        if self.low_level_policy_type =='IK':
            self.policy.last_des_body_ori = np.array([0, 0, state['base_ori_euler'][2]])

    def load_model(self, save_dir):
        if self.low_level_policy_type =='NN':
            self.policy.load_state_dict(
                torch.load('%s/NNTG.pt' % (save_dir)))

    