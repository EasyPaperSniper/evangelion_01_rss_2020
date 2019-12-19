# This file is for high level planning
# Contains different sampling methods
#

import math
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

NUM_LEGS = 6
TRIPOD_LEG_PAIR_1 = [0, 3, 4]
TRIPOD_LEG_PAIR_2 = [1, 2, 5]
class forward_model(nn.Module):
    '''
    The forward model(mid-level policy) is a NN which is trained in a supervised manner
    '''
    def __init__(self,model_obs_dim, z_dim, model_output_dim, model_hidden_num, device):
        '''
        Initialize the structure and options for model
        '''
        super().__init__()
        self.device = device

        self.trunk = nn.Sequential(
            nn.Linear(model_obs_dim + z_dim, model_hidden_num ), nn.ReLU(),
            nn.Linear(model_hidden_num, model_hidden_num), nn.ReLU(),
            nn.Linear(model_hidden_num, model_output_dim))

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
    #TODO: add planning part........

# class curiosity_policy():
#     def __init__(self):

#     def sample_latent_action(self):

#     def update_policy(self):

class raibert_footstep_policy():
    def __init__(self, 
                stance_duration = 50,
                target_speed = np.array([0.0,0.2]),
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
    
    def plan_latent_action(self,state):
        latent_action = np.zeros(13)
        current_speed = state['base_velocity'][0:2]
        speed_term = self.stance_duration/(2*self.control_frequency) * self.target_speed #current_speed
        acceleration_term = self.speed_gain *(current_speed - self.target_speed)
        
        # X = T/2 * x_dot + k_p (x_dot - x_dot_des)
        des_footstep = (speed_term + acceleration_term)
        for i in range(6):
            if i in self.stance_set:
                latent_action[i*2:i*2+2] = -des_footstep
            else:
                latent_action[i*2:i*2+2] = des_footstep
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
    def __init__(self, z_dim, limits, low_level_policy_type='IK', sample_num = 50, predict_horizon = 1):
        '''
        z_dim: dimension of the latent action
        scale: the scale of variance in different dim 
        '''
        self.z_dim = z_dim
        self.limits = limits
        self.low_level_policy_type = low_level_policy_type
        self.sample_num = sample_num
        self.predict_horizon = predict_horizon

    def sample_latent_action(self):
        action = np.clip(np.random.randn(self.z_dim), -1,1)
        if self.low_level_policy_type =='IK':
            for i in range(0, self.z_dim-1, 2):
                action[i] = action[i] * self.limits[0]
                action[i+1] = action[i+1] * self.limits[1]
            action[-1] = action[i] * 0.05 * math.pi

        return action
    
    def plan_latent_action(self, state, model, cost_func, p, sample_num = 30, horizon = 1):
        '''
        Sample action based on the reward function
        input:
            state: dict
            model: the NN model 
            reward_func: function
        output:
            latent action: np.array(self.z_dim)
        '''
        # sample bunch of actions and plan for single step
        latent_action_buffer = np.empty([sample_num, horizon,self.z_dim])
        for sample_index in range(sample_num):
            for horizon_index in range(horizon):
                latent_action_buffer[sample_index][horizon_index] = self.sample_latent_action()

        # run model to calculate reward
        cost = [p.apply(utils.run_mpc,args=(state, model, reward_func, latent_action_sample)) 
                                        for latent_action_sample in latent_action_buffer]
        
        # return best reward index and select action
        sqe_index = cost.index(min(cost))
        latent_action = latent_action_buffer[sqe_index][0]

        return latent_action

class high_level_planning():
    def __init__(self,
        device,
        model_obs_dim,
        z_dim,
        model_output_dim,
        model_hidden_num,
        batch_size = 64,
        model_lr = 1e-4,
        high_level_policy_type = 'random',
        update_sample_policy = 0,
        update_sample_policy_lr = 1e-3,
        num_timestep_per_footstep = 50,
        low_level_policy_type = 'IK',
        model_update_steps = 10,
        **kwargs
        ):
        # Initialize model & sampling policy & buffer
        self.update_step = 0
        self.model_obs_dim = model_obs_dim
        self.z_dim = z_dim
        self.model_output_dim = model_output_dim
        self.model_hidden_num = model_hidden_num
        self.model_update_steps= model_update_steps
        self.device = device


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
            self.limits = np.array([0.15, 0.15])
        else:
            self.limits = np.ones(z_dim)

        self.batch_size = batch_size

        self.forward_model =  forward_model(model_obs_dim, z_dim, model_output_dim, model_hidden_num, device)
        self.model_lr = model_lr
        self.model_optimizer = torch.optim.Adam(self.forward_model.parameters(),lr=self.model_lr)

        if high_level_policy_type == 'random':
            self.policy = random_policy(z_dim, self.limits, low_level_policy_type)
            self.update_sample_policy = False
        elif high_level_policy_type == 'raibert':
            self.policy = raibert_footstep_policy(stance_duration = num_timestep_per_footstep)
            self.update_sample_policy = False
        else:
            print('Not implement yet!!')
            # self.policy = curiosity_policy()
            # self.update_sample_policy = True
            # self.policy_lr = update_sample_policy_lr
            # self.policy_optimizer = torch.optim.Adam(self.policy.parameters(),lr=self.policy_lr)
        

    def update_model(self, HL_replay_buffer):
        for _ in range(self.model_update_steps):
            self.update_step += 1

            idxs = np.random.randint(
                0, HL_replay_buffer.capacity if HL_replay_buffer.full else HL_replay_buffer.idx, size=self.batch_size)

            state = torch.as_tensor(utils.normalization(HL_replay_buffer.obses[idxs], self.all_mean_var[0], self.all_mean_var[1]), device=self.device).float()
            action = torch.as_tensor(utils.normalization(HL_replay_buffer.actions[idxs], self.all_mean_var[2], self.all_mean_var[3]), device=self.device)
            delta_state = torch.as_tensor(
                utils.normalization(HL_replay_buffer.next_obses[idxs], self.all_mean_var[4], self.all_mean_var[5]), device=self.device).float()

            pred_delta_state = self.forward_model(state,action)
            model_loss = F.mse_loss(pred_delta_state, delta_state)
            self.model_optimizer.zero_grad()
            model_loss.backward()
            self.model_optimizer.step()


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


    def sample_latent_action(self):
        latent_action = self.policy.sample_latent_action()
        return latent_action

    def update_policy(self):
        if self.update_sample_policy:
            self.policy.update_policy()

    def plan_latent_action(self,state,cost_func =None):
        # self.p = mp.Pool(mp.cpu_count())
        # return self.policy.plan_latent_action(state, self.forward_model, reward_func, self.p)
        return self.policy.plan_latent_action(state)

    def save_data(self,save_dir):
        torch.save(self.forward_model.state_dict(),
                   '%s/model.pt' % (save_dir) )

    def load_data(self, save_dir):
        self.forward_model.load_state_dict(
            torch.load('%s/model.pt' % (save_dir)))

    def load_mean_var(self,save_dir):
        '''
        load normalization parameters(mean, var) for forward model
        0: mean of obses; 1: var of obses; 2: mean of actions; 3: var of actions; 4: mean of next_obses; 5: var of next_obses
        '''
        self.all_mean_var = np.load(save_dir+'/all_mean_var.npy')
        return self.all_mean_var