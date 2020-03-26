# This file is for high level planning
# Contains different sampling methods
#

import math
import multiprocessing as mp
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from baseline_code import SAC
from NN_learned_action import learned_z
from pytorchtools.pytorchtools import EarlyStopping

NUM_LEGS = 6
TRIPOD_LEG_PAIR_1 = [0, 3, 4]
TRIPOD_LEG_PAIR_2 = [1, 2, 5]

class forward_model(nn.Module):
    '''
    The forward model(mid-level policy) is a NN which is trained in a supervised manner
    '''
    def __init__(self,model_obs_dim, z_dim, model_output_dim, model_hidden_num, model_layer_num,device):
        '''
        Initialize the structure and options for model
        '''
        super().__init__()
        self.device = device
        modules = []
        modules.append(nn.Linear(model_obs_dim + z_dim,512))
        modules.append(nn.ReLU())
        # # for _ in range(1,model_layer_num):
        # modules.append(nn.Linear(16,16))
        # modules.append(nn.ReLU())
        modules.append(nn.Linear(512, 512))
        modules.append(nn.ReLU())
        # modules.append(nn.Linear(256, 16))
        # modules.append(nn.ReLU())
        modules.append(nn.Linear(512, model_output_dim))
        self.trunk = nn.Sequential(*modules)


    def forward(self,model_obs, latent_action):
        obs_action = torch.cat([model_obs, latent_action], dim=1)
        return self.trunk(obs_action)

    def predict(self,model_obs, latent_action):
        model_obs = torch.FloatTensor(model_obs).to(self.device)
        model_obs = model_obs.unsqueeze(0)
        latent_action = torch.FloatTensor(latent_action).to(self.device)
        latent_action = latent_action.unsqueeze(0)
        prediction = self.forward(model_obs, latent_action)
        return prediction.cpu().data.numpy().flatten()

    def predict_para(self, model_obs, latent_action):
        model_obs = torch.as_tensor(model_obs, device = self.device).float()
        latent_action = torch.as_tensor(latent_action, device = self.device).float()
        prediction = self.forward(model_obs, latent_action)
        return prediction.cpu().data.numpy()


# class curiosity_policy():
#     def __init__(self):

#     def sample_latent_action(self):

#     def update_policy(self):

class raibert_footstep_policy():
    def __init__(self, 
                stance_duration = 50,
                target_speed = np.array([0.0,0.0,0.1]),
                speed_gain = 0.0,
                des_body_ori = [0,0,0],
                control_frequency = 60):
        self.stance_duration = stance_duration
        self.speed_gain = speed_gain
        self.target_speed = target_speed
        self.des_body_ori = des_body_ori
        self.control_frequency = control_frequency
        self.swing_set = TRIPOD_LEG_PAIR_1
        self.stance_set = TRIPOD_LEG_PAIR_2 
    
    def plan_latent_action(self,state, target_speed=None):
        latent_action = np.zeros(3)
        current_speed = state['base_velocity'][0:2]
        # if target_speed.any():
        self.target_speed = target_speed
        
        speed_term = self.stance_duration/(2*self.control_frequency) * self.target_speed[0:2] #current_speed
        acceleration_term = self.speed_gain *(current_speed - self.target_speed[0:2])
        orientation_speed_term = -self.stance_duration/(self.control_frequency)*  self.target_speed[2] # state['base_ori_euler'][2]#

        # X = T/2 * x_dot + k_p (x_dot - x_dot_des)
        des_footstep = (speed_term + acceleration_term)
        latent_action[0:2] = des_footstep
        latent_action[2] = orientation_speed_term
        self.swing_set, self.stance_set = np.copy(self.stance_set), np.copy(self.swing_set)
        return latent_action

    def sample_latent_action(self):
        return False

# class CEM_planning():
#     def __init__():

#     def plan_latent_action(self, state):
        
#         return latent_action


class random_policy():
    '''
    The policy is defined in the polar coordinate (r, theta)
    '''
    def __init__(self, z_dim, limits, low_level_policy_type='IK',stance_duration = 50, control_frequency = 60):
        '''
        z_dim: dimension of the latent action
        scale: the scale of variance in different dim 
        '''
        self.z_dim = z_dim
        self.limits = limits
        self.low_level_policy_type = low_level_policy_type
        self.stance_duration = stance_duration
        self.control_frequency = control_frequency


    def sample_latent_action(self):
        if self.low_level_policy_type =='NN':
            # total_num = np.shape(learned_z)[0]
            # index = np.random.randint(0, total_num, size =1)[0]
            # action =  np.array(learned_z[index]) + np.clip(0.3*np.random.randn(self.z_dim),-1.0,1.0)
            action = np.clip(1.3 * np.random.randn(self.z_dim),-3,3) 
            return action

        if self.low_level_policy_type =='IK':
            action = 0.1 * np.clip( np.random.randn(self.z_dim), -1.5,1.5)
            action[2] = action[2] * 1.5
                
        return action
    
    def plan_latent_action(self, state):
        '''
        Sample action based on the reward function
        input:
            target_speed
        output:
            latent action: np.array(self.z_dim)
        '''
        if self.low_level_policy_type =='NN':
            return self.sample_latent_action()
        
        latent_action = np.clip(np.random.randn(self.z_dim), -2,2) * 0.1 #+ target_speed * 0.05
        # latent_action[2] = -state['base_ori_euler'][2]
        return self.sample_latent_action()

class high_level_planning():
    def __init__(self,
        device,
        model_obs_dim,
        z_dim,
        model_output_dim,
        model_hidden_num,
        model_layer_num,
        batch_size = 64,
        model_lr = 1e-4,
        high_level_policy_type = 'random',
        update_sample_policy = 0,
        update_sample_policy_lr = 1e-3,
        num_timestep_per_footstep = 50,
        low_level_policy_type = 'IK',
        model_update_steps = 10,
        control_frequency = 60,
        **kwargs
        ):
        # Initialize model & sampling policy & buffer
        self.update_step = 0
        self.model_obs_dim = model_obs_dim
        self.z_dim = z_dim
        self.model_output_dim = model_output_dim
        self.model_layer_num = model_layer_num
        self.model_hidden_num = model_hidden_num
        self.model_update_steps= model_update_steps
        self.device = device
        self.high_level_policy_type = high_level_policy_type
        self.num_timestep_per_footstep = num_timestep_per_footstep
        self.control_frequency = control_frequency

        # normalization parameter for model input/output
        self.all_mean_var = np.array([
            np.zeros(model_obs_dim),
            np.ones(model_obs_dim),
            np.zeros(z_dim),
            np.ones(z_dim),
            np.zeros(model_output_dim),
            np.ones(model_output_dim),
        ])


        if low_level_policy_type == 'IK':
            self.limits = np.array([0.1, 0.1])
        else:
            self.limits = np.ones(z_dim)

        self.batch_size = batch_size

        self.forward_model =  forward_model(model_obs_dim, z_dim, model_output_dim, model_hidden_num,model_layer_num, device)
        self.model_lr = model_lr
        self.model_optimizer = torch.optim.Adam(self.forward_model.parameters(),lr=self.model_lr, weight_decay=2e-4)

        if high_level_policy_type == 'random':
            self.policy = random_policy(z_dim, self.limits, low_level_policy_type, stance_duration  = num_timestep_per_footstep, control_frequency = control_frequency)
            self.update_sample_policy = False
        elif high_level_policy_type == 'raibert':
            self.policy = raibert_footstep_policy(stance_duration  = num_timestep_per_footstep, control_frequency = control_frequency)
            self.update_sample_policy = False
        elif high_level_policy_type == 'SAC':
            # defining SAC here
            self.update_sample_policy = update_sample_policy
            self.policy = SAC(device = self.device,
                            obs_dim = self.model_obs_dim,
                            action_dim = self.z_dim,
                            hidden_dim = self.model_hidden_num,   
                            actor_lr=update_sample_policy_lr,
                            critic_lr=update_sample_policy_lr,)

        else:
            print('Not implement yet!!')
            # self.policy = curiosity_policy()
            # self.update_sample_policy = True
            # self.policy_lr = update_sample_policy_lr
            # self.policy_optimizer = torch.optim.Adam(self.policy.parameters(),lr=self.policy_lr)
        

    def update_model(self, HL_replay_buffer, logger):
        early_stopper = EarlyStopping(patience=7)
        split = 10.0
        state_norm = utils.normalization(HL_replay_buffer.obses, self.all_mean_var[0], self.all_mean_var[1])
        action_norm = utils.normalization(HL_replay_buffer.actions, self.all_mean_var[2], self.all_mean_var[3])
        delta_state_norm = utils.normalization(HL_replay_buffer.next_obses, self.all_mean_var[4], self.all_mean_var[5])
        train_capacity = int(HL_replay_buffer.capacity * (split-1)/split)
        test_idxs = np.arange(-int(HL_replay_buffer.capacity / split) ,0)

        state_test = torch.as_tensor(state_norm[test_idxs], device=self.device).float()
        action_test = torch.as_tensor(action_norm[test_idxs], device=self.device).float()
        delta_state_test = torch.as_tensor(delta_state_norm[test_idxs], device=self.device).float()

        for i in range(self.model_update_steps):
            self.update_step += 1
            idxs = np.random.randint( 0, train_capacity , size=self.batch_size)
            # idxs = np.random.randint(0, 1100, size=self.batch_size)

            state = torch.as_tensor(state_norm[idxs], device=self.device).float()
            action = torch.as_tensor(action_norm[idxs], device=self.device).float()
            delta_state = torch.as_tensor(delta_state_norm[idxs], device=self.device).float()

            pred_delta_state = self.forward_model(state,action)
            model_loss = F.mse_loss(pred_delta_state, delta_state)
            self.model_optimizer.zero_grad()
            model_loss.backward()
            self.model_optimizer.step()

            logger.log('train/model_loss', model_loss)
            logger.dump(self.update_step)

            if (i+1) %100==0:
                pred_delta_state = self.forward_model(state_test,action_test)
                model_loss = F.mse_loss(pred_delta_state, delta_state_test)
                logger.log('train/val_loss', model_loss)
                logger.dump(self.update_step)
                early_stopper(model_loss)

            if early_stopper.early_stop:
                break

        self.save_data('.')

    def update_model_norm(self, all_mean_var):
        '''
        update normalization parameters(mean, var) for forward model
        '''
        self.all_mean_var = all_mean_var

    def model_predict(self, model_obs, latent_action):
        model_obs_norm = utils.normalization(model_obs, mean= self.all_mean_var[0], std = self.all_mean_var[1])
        latent_action_norm = utils.normalization(latent_action, mean = self.all_mean_var[2], std = self.all_mean_var[3])
        predict_norm = self.forward_model.predict(model_obs_norm, latent_action_norm)
        return utils.inverse_normalization(predict_norm, mean = self.all_mean_var[4], std = self.all_mean_var[5])


    def sample_latent_action(self,state=None, target=None, com_utils=None):
        if self.high_level_policy_type =='SAC':
            obs = com_utils.HL_obs(state, target)
            return self.policy.sample_latent_action(obs)
        latent_action = self.policy.sample_latent_action()
        return latent_action

    def update_policy(self, replay_buffer):
        if self.update_sample_policy:
            for _ in range(self.update_sample_policy):
                self.policy.update_policy(replay_buffer)
        self.save_data('.')


    def plan_latent_action(self, state, target, com_utils=None, sample_num = 1500, horizon = 2):
        if self.high_level_policy_type =='raibert':
            self.policy.target_speed = target
            return self.policy.plan_latent_action(state,target)

        if self.high_level_policy_type =='SAC':
            obs = com_utils.HL_obs(state, target)
            return self.policy.plan_latent_action(obs)
        
        if self.high_level_policy_type in ['random']:
            # sample bunch of actions and plan for single step
            latent_action_buffer = np.empty([horizon, sample_num, self.z_dim])
            HL_obs_buffer = np.empty([sample_num, self.model_obs_dim])
            position_buffer = np.empty([sample_num, 5])
            cost = np.zeros(sample_num)
            HL_obs = com_utils.HL_obs(state)

            for sample_index in range(sample_num):
                HL_obs_buffer[sample_index] = HL_obs
                position_buffer[sample_index]= np.array([state['base_ori_euler'][2],state['base_pos_x'][0], state['base_pos_y'][0], 
                                                            state['base_velocity'][0],state['base_velocity'][1]])
                for horizon_index in range(horizon):
                    latent_action_buffer[horizon_index][sample_index] = self.policy.plan_latent_action(state)
            
            cost = com_utils.run_mpc_calc_cost(HL_obs_buffer, self.forward_model, target, latent_action_buffer, position_buffer, cost, self.all_mean_var).tolist()

            # return best reward index and select action
            sqe_index = cost.index(min(cost))
            latent_action = latent_action_buffer[0][sqe_index]
            return latent_action
    
    
    def save_data(self,save_dir):
        if self.update_sample_policy:
            self.policy.save(save_dir)
        torch.save(self.forward_model.state_dict(),
                   '%s/model.pt' % (save_dir) )

    def load_data(self, save_dir):
        if self.update_sample_policy:
            self.policy.load(save_dir)
        self.forward_model.load_state_dict(
            torch.load('%s/model.pt' % (save_dir)))

    def load_mean_var(self,save_dir):
        '''
        load normalization parameters(mean, var) for forward model
        0: mean of obses; 1: var of obses; 2: mean of actions; 3: var of actions; 4: mean of next_obses; 5: var of next_obses
        '''
        self.all_mean_var = np.load(save_dir+'/all_mean_var.npy')
        return self.all_mean_var