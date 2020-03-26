import os
import json
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from joystick_test import expert_control, expert_control_back
import matplotlib.pyplot as plt
from daisy_API import daisy_API
import daisy_hardware.motion_library as motion_library

from logger import Logger
import utils
from low_level_traj_gen import NN_tra_generator

# # load trajectory


class train_NNTG():
    def __init__(self,  
                num_primitive, 
                z_dim,
                policy_output_dim, 
                policy_hidden_num, 
                policy_lr, 
                batch_size,
                device):
            
        self.policy = NN_tra_generator(z_dim, policy_output_dim, policy_hidden_num, device)
        self.policy_lr = policy_lr
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(),lr=self.policy_lr)
        self.batch_size = batch_size

        self.num_primitive = num_primitive
        self.z_dim = z_dim
        self.device = device
        self.learning_step = 0

        # define random z_action
        self.z_action_all = torch.tensor(np.random.normal(0,0.2,(num_primitive,z_dim)).tolist(),requires_grad=True, device = device)
        self.z_action_optimizer = torch.optim.Adam([self.z_action_all],lr=self.policy_lr, weight_decay=0.01)


    def sample_phase_action(self, primitive_index):
        idxs = np.random.randint(1,171, size=self.batch_size)
        phase = idxs / 170.0
        expert_action = np.empty((self.batch_size, 18), dtype=np.float32)
        
        for i in range(self.batch_size):
            expert_action[i] = traj[primitive_index][idxs[i]-1]

        return torch.as_tensor(np.reshape(phase,(self.batch_size,1)), device= self.device).float(), torch.as_tensor(expert_action, device= self.device).float()


    def update_model(self, num_iteration, save_dir):
        logger = Logger(save_dir, name = 'train')
        for _ in range(num_iteration):
            for i in range(self.num_primitive):
                # sample phase and expert action

                phase_vec, expert_action = self.sample_phase_action(i)
                z_vec = self.z_action_all[i] * torch.tensor(np.ones((self.batch_size,self.z_dim)).tolist(), requires_grad= True, device = self.device)

                pred_action = self.policy(z_vec, phase_vec)

                policy_loss = F.mse_loss(pred_action, expert_action) 
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                self.z_action_optimizer.step()

                self.learning_step += 1
                logger.log('train/model_loss', policy_loss)
                logger.dump(self.learning_step)
        
        self.save_model(save_dir)

    def save_model(self, save_dir):
        torch.save(self.policy.state_dict(),
                   '%s/NNTG.pt' % (save_dir) )
    
    def load_model(self, save_dir):
        self.policy.load_state_dict(
            torch.load('%s/NNTG.pt' % (save_dir)))



z_dim = 2
num_primitive = 10
policy_output_dim = 18
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tra_learning = train_NNTG( num_primitive = num_primitive, 
            z_dim= z_dim,
            policy_output_dim = policy_output_dim, 
            policy_hidden_num =512, 
            policy_lr = 1e-3, 
            batch_size = 12,
            device = device)
mean_std = np.load('./save_data/trial_'+str(z_dim) +'/LL_mean_std.npy')

z_action_all = np.clip(1*np.random.randn(80,z_dim),-0.8,0.8)
z_action_all =[
        [ 3.5490, -3.6426],
        [-3.7186, -3.7996],
        [ 0,     -3.70]
        ]
z_action_all = np.ones((7*7,z_dim))

# print(min(z_action_all), max(z_action_all))
total_step = 5
traj = np.load('./save_data/expert_action_total.npy')
CoM_traj = np.empty((np.shape(z_action_all)[0], 5, total_step*2))
action_record = np.empty((np.shape(z_action_all)[0],  100,18))
gait_record = np.empty((np.shape(z_action_all)[0], 6, 100))
tra_learning.load_model(save_dir = './save_data/trial_'+str(z_dim))
env = daisy_API(sim=True, render=True, logger = False)
env.set_control_mode('position')
state = env.reset()


for i in range(np.shape(z_action_all)[0]):
    state = motion_library.exp_standing(env)
    first = int(i/7.0)
    second = i%7
    
    z_action_all[i][0] = -3 + first
    z_action_all[i][1] = -3 + second
    best_z_action = z_action_all[i]
    # min_loss = 10000
    # for j in range(10000):
    #     cur_z_action  = 3*np.random.randn(3)
    #     tra_learning.policy.z_action = cur_z_action
    #     action_all = np.zeros((100,18))
    #     for k in range(100):
    #         action_all[k] = tra_learning.policy.get_action(state, (k+1)/100.0)
    #     loss = np.linalg.norm(utils.normalization(traj, mean_std[0], mean_std[1]) - action_all)
    #     if loss< min_loss:
    #         min_loss=loss
    #         best_z_action = cur_z_action
        
    tra_learning.policy.z_action = best_z_action
    for j in range(total_step):
        CoM_traj[i][0][2*j] = state['base_pos_x'][0]
        CoM_traj[i][1][2*j] = state['base_pos_y'][0]
        CoM_traj[i][2][2*j] = state['base_ori_euler'][2]
        CoM_traj[i][3][2*j] = state['base_velocity'][0]
        CoM_traj[i][4][2*j] = state['base_velocity'][1]
        
        for k in range(100):
            action = tra_learning.policy.get_action(state, (k+1)/100.0)
            action = utils.inverse_normalization(action, mean_std[0], mean_std[1])
            if j == total_step - 1:
                action_record[i][k] = action
            state = env.step(action)
            if k == 49:
                CoM_traj[i][0][2*j+1] = state['base_pos_x'][0]
                CoM_traj[i][1][2*j+1] = state['base_pos_y'][0]
                CoM_traj[i][2][2*j+1] = state['base_ori_euler'][2]
                CoM_traj[i][3][2*j+1] = state['base_velocity'][0]
                CoM_traj[i][4][2*j+1] = state['base_velocity'][1]
            
            # if j == 8:
            #     for q in range(6):
            #         if state['foot_pos'][q][2]<=-0.005:
            #             gait_record[i][q][k] = 1
            #         else:
            #             gait_record[i][q][k] = 0


# np.save('./save_data/trial_2/500_random_gait_record.npy', gait_record)


np.save('./save_data/trial_'+str(z_dim)+'/CoM_traj.npy', CoM_traj)
np.save('./save_data/trial_'+str(z_dim)+'/learned_action.npy', action_record)
# np.save('./save_data/trial_2/gait_record.npy', gait_record)
