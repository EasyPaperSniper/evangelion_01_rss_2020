# This file is for high level planning
# Contains different sampling methods
#

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class forward_model(nn.Module):
    '''
    The forward model(mid-level policy) is a NN which is trained in a supervised manner
    '''
    def __init__(self,HL_obs_dim, z_dim, HL_output_dim, model_hidden_num):
        '''
        Initialize the structure and options for model
        '''
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(HL_obs_dim + z_dim, model_hidden_num ), nn.ReLU(),
            nn.Linear(model_hidden_num, model_hidden_num), nn.ReLU(),
            nn.Linear(model_hidden_num, HL_output_dim))

    def forward(self,HL_obs, latent_action):
        obs_action = torch.cat([HL_obs, latent_action], dim=1)
        return self.trunk(obs_action)



# class curiosity_policy():
#     def __init__(self):

#     def sample_latent_action(self):

#     def update_policy(self):


class random_policy():
    '''
    The policy is defined in the polar coordinate (r, theta)
    '''
    def __init__(self, z_dim, limits):
        '''
        z_dim: dimension of the latent action
        scale: the scale of variance in different dim 
        '''
        self.z_dim = z_dim
        self.limits = limits

    def sample_latent_action(self):
        action = np.random.randn(self.z_dim)
        for i in range(0, self.z_dim, 2):
            action[i] = action[i] * self.limits[0]
            action[i+1] = action[i+1] * self.limits[1]
        return action



class high_level_planning():
    def __init__(self,
        HL_obs_dim,
        z_dim,
        HL_output_dim,
        model_hidden_num,
        limits,
        batch_size,
        sample_num,
        model_lr,
        policy_type = 'random',
        **kwargs
        ):
        # Initialize model & sampling policy & buffer
        self.HL_obs_dim = HL_obs_dim
        self.z_dim = z_dim
        self.HL_output_dim = HL_output_dim
        self.model_hidden_num = model_hidden_num
        self.limits = limits

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.forward_model =  forward_model(HL_obs_dim, z_dim, HL_output_dim, model_hidden_num)
        self.model_lr = model_lr
        self.model_optimizer = torch.optim.Adam(self.forward_model.parameters(),lr=self.model_lr)

        if policy_type == 'random':
            self.policy = random_policy(z_dim, self.limits)
            self.update_sample_policy = False
        else:
            self.update_sample_policy = True
            self.policy_lr = policy_lr
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(),lr=self.policy_lr)
        

    def update_model(self, HL_replay_buffer):
        state, action,_, delta_state,_ = HL_replay_buffer.sample(self.batch_size)
        pred_delta_state = self.forward_model(state,action)
        model_loss = F.mse_loss(pred_delta_state, delta_state)

        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()
        
    def saturation_func(self, action):
        '''
        Saturate the action
        Input:
            action: latent action
        '''
        for i in range(0, self.z_dim, 2):
            action[i] = max(min(self.limits[0], action[i]), -self.limits)
            action[i+1] = max(min(self.limits[1], action[i+1]), -self.limits)
        return action

    def sample_latent_action(self):
        latent_action = self.saturation_func(self.policy.sample_latent_action())
        return latent_action, 

    def update_policy(self):
        if self.update_sample_policy:
            self.policy.update_policy()

    def plan_latent_action(self):
        # multi processing
        return True

    def save_model(self):
        return 

    def load_model(self):
        return 
