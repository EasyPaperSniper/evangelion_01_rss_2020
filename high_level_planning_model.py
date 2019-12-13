# This file is for high level planning
# Contains different sampling methods
#

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
                speed_gain = 0.2,
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
    def __init__(self, z_dim, limits, low_level_policy_type='IK'):
        '''
        z_dim: dimension of the latent action
        scale: the scale of variance in different dim 
        '''
        self.z_dim = z_dim
        self.limits = limits
        self.low_level_policy_type = low_level_policy_type

    def sample_latent_action(self):
        action = np.clip(np.random.randn(self.z_dim), -1,1)
        if self.low_level_policy_type =='IK':
            for i in range(0, self.z_dim-1, 2):
                action[i] = action[i] * self.limits[0]
                action[i+1] = action[i+1] * self.limits[1]
            action[-1] = action[i] * 0.05 * math.pi

        return action
    
    def plan_latent_action(self):
        return self.sample_latent_action()

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
        update_per_iter = 10,
        **kwargs
        ):
        # Initialize model & sampling policy & buffer
        self.update_step = 0
        self.model_obs_dim = model_obs_dim
        self.z_dim = z_dim
        self.model_output_dim = model_output_dim
        self.model_hidden_num = model_hidden_num
        self.update_per_iter = update_per_iter
        self.device = device

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
        

    def update_model(self, HL_replay_buffer, logger):
        for _ in range(self.update_per_iter):
            self.update_step += 1
            state, action,_, delta_state,_ = HL_replay_buffer.sample(self.batch_size)
            pred_delta_state = self.forward_model(state,action)
            model_loss = F.mse_loss(pred_delta_state, delta_state)

            self.model_optimizer.zero_grad()
            model_loss.backward()
            self.model_optimizer.step()
            logger.log('train/model_loss', model_loss)
        logger.dump(self.update_step,console = True)
        
    def sample_latent_action(self):
        latent_action = self.policy.sample_latent_action()
        return latent_action

    def update_policy(self):
        if self.update_sample_policy:
            self.policy.update_policy()

    def plan_latent_action(self,state):
        return self.policy.plan_latent_action(state)

    def save_data(self,save_dir):
        torch.save(self.forward_model.state_dict(),
                   '%s/model.pt' % (save_dir) )

    def load_data(self, save_dir):
        self.forward_model.load_state_dict(
            torch.load('%s/model.pt' % (save_dir)))
