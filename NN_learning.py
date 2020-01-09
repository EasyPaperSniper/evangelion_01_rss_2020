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
        idxs = np.random.randint(1, 101, size=self.batch_size)
        phase = idxs / 100.0
        expert_action = np.empty((self.batch_size, 18), dtype=np.float32)
        
        for i in range(self.batch_size):
            if primitive_index == 0:
                expert_action[i] = expert_control(i= phase[i],w = 1, phase=1)
            elif primitive_index == 1:
                expert_action[i] = expert_control(i= phase[i],w = 1, phase=0.5)
            elif primitive_index == 2:
                expert_action[i] = expert_control(i= phase[i],w = 1, phase=1.5)
            elif primitive_index == 3:
                expert_action[i] = expert_control(i= phase[i],a = 0.6, w = 1, phase=1)
            else:
                expert_action[i] = expert_control_back(i= phase[i],w = 1, phase=1)

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


z_dim = 1
num_primitive = 5
policy_output_dim = 18
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tra_learning = train_NNTG( num_primitive = num_primitive, 
            z_dim= z_dim,
            policy_output_dim = policy_output_dim, 
            policy_hidden_num = 32, 
            policy_lr = 1e-3, 
            batch_size = 10,
            device = device)




# # train NNTG
# print('Z_action before optimization', tra_learning.z_action_all)
# tra_learning.update_model(num_iteration =1000, save_dir = './save_data/trial_2')
# print('Z_action after optimization', tra_learning.z_action_all)
# [[-0.1715],
#         [ 0.0839],
#         [-0.0490],
#         [-0.3888],
#         [ 0.4106]]




# # test NNTG
# test_num = 5
# tra_learning.load_model(save_dir = './save_data/trial_2')
# record_data = np.empty((test_num,100))
# z_action = np.random.normal(0.0, 0.5, (test_num,1))
# axis = range(100)

# for i in range(100):
#     for j in range(test_num):
#         record_data[j][i] = tra_learning.policy.get_action(z_action[j], np.array([((i+1)%100)/100.0]))[0]

# for i in range(test_num):
#     plt.plot(axis, record_data[i], label = 'z= %s'%(str(z_action[i])))
# plt.legend()

# plt.show()





# test in simulation 
# test_length = 1000
# z = 0.4106
# tgt_com_tra = np.empty((3,test_length))
# test_com_tra = np.empty((3, test_length))
# env = daisy_API(sim=True, render=True, logger = False)
# env.set_control_mode('position')

# # state = motion_library.exp_standing(env, shoulder=0.3, elbow = 1.3)
# # init_action = state['j_pos']
# # for i in range(test_length):
# #     action = expert_control(i= ((i+1)%50)/50.0 ,w = 1, phase=0.5) + init_action
# #     state = env.step(action)
# #     tgt_com_tra[0][i] = state['base_pos_x'][0]
# #     tgt_com_tra[1][i] = state['base_pos_y'][0]
# #     tgt_com_tra[2][i] = state['base_ori_euler'][2]

# state = motion_library.exp_standing(env, shoulder=0.3, elbow = 1.3)
# init_action = state['j_pos']
# for i in range(test_length):
#     action = tra_learning.policy.get_action(np.array([z]), np.array([((i+1)%50)/50.0])) + init_action
#     state = env.step(action)
#     test_com_tra[0][i] = state['base_pos_x'][0]
#     test_com_tra[1][i] = state['base_pos_y'][0]
#     test_com_tra[2][i] = state['base_ori_euler'][2]

# np.save('./save_data/test_tra/expert_tra.npy', tgt_com_tra)
# np.save('./save_data/test_tra/test_tra.npy', test_com_tra)

# tgt_com_tra = np.load('./save_data/test_tra/expert_tra.npy')
# test_com_tra = np.load('./save_data/test_tra/test_tra.npy')

# axis = range(np.shape(tgt_com_tra)[1])
    
# plt.rcParams['figure.figsize'] = (8, 10)
# fig, (ax1,ax2,ax3) = plt.subplots(3,1)
# ax1.plot(axis, tgt_com_tra[0])
# ax1.plot(axis, test_com_tra[0])
# ax1.set_title('X tracking')
# ax2.plot(axis, tgt_com_tra[1])
# ax2.plot(axis, test_com_tra[1])
# ax2.set_title('Y tracking')
# ax3.plot(axis, tgt_com_tra[2])
# ax3.plot(axis, test_com_tra[2])
# ax3.set_title('yaw tracking')
# plt.show()