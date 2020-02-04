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
traj = np.load('./save_data/expert_trajectory_total.npy')
tra_learning = train_NNTG( num_primitive = num_primitive, 
            z_dim= z_dim,
            policy_output_dim = policy_output_dim, 
            policy_hidden_num = 512, 
            policy_lr = 1e-3, 
            batch_size = 12,
            device = device)

# z_action_all = np.arange(-0.5,0.5,0.1)
z_action_all = np.random.randn(40,z_dim)
total_step = 10
z_action_all = [[ 0.2295,  0.5950],
        [ 0.1303,  0.0111],
        [-0.0920,  0.3827],
        [ 0.2068,  0.1836],
        [ 0.2663, -0.0928],
        [ 0.2584,  0.1057],
        [ 0.2051, -0.3227],
        [ 0.3041,  0.1780],
        [ 0.5115,  0.1338],
        [ 0.0178,  0.1417],
        [-0.3598, -0.3446],
        [-0.2903, -0.1586],
        [-0.3960, -0.3700],
        [-0.1368, -0.1423],
        [-0.5574, -0.2970],
        [-0.3518,  0.2172],
        [-0.1867, -0.6132],
        [-0.1863, -0.2268],
        [ 0.5526, -0.3470],
        [-0.2774,  0.1772]]
CoM_traj = np.empty((np.shape(z_action_all)[0], 3, total_step))
gait_record = np.empty((np.shape(z_action_all)[0], 6, 100))
tra_learning.load_model(save_dir = './save_data/trial_2')
env = daisy_API(sim=True, render=False, logger = False)
env.set_control_mode('position')
state = env.reset()


for i in range(np.shape(z_action_all)[0]):
    state = motion_library.exp_standing(env, shoulder = 0.3, elbow = 1.3)
    
    
    z_action = np.array(z_action_all[i])
    tra_learning.policy.z_action = z_action
    for j in range(total_step):
        CoM_traj[i][0][j] = state['base_pos_x'][0]
        CoM_traj[i][1][j] = state['base_pos_y'][0]
        CoM_traj[i][2][j] = state['base_ori_euler'][2]
        
        for k in range(100):
            action = tra_learning.policy.get_action(state, (k+1)/100.0)
            state = env.step(action)
            
            # if j == 8:
            #     for q in range(6):
            #         if state['foot_pos'][q][2]<=-0.005:
            #             gait_record[i][q][k] = 1
            #         else:
            #             gait_record[i][q][k] = 0


# np.save('./save_data/trial_2/500_random_gait_record.npy', gait_record)


np.save('./save_data/trial_2/CoM_traj.npy', CoM_traj)
# np.save('./save_data/trial_2/gait_record.npy', gait_record)


# traj = np.load('./save_data/trajectories_gaits.npz')
# traj_1 = traj['traj_good']
# for i in [11,2,13,5,14,7,4,8,9,]:
#     input(str(i)+ ' iteration')
#     state = motion_library.exp_standing(env, shoulder=0.3, elbow = 1.3)
#     for j in range(400):
#         env.step(traj_1[i][j])