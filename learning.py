import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_action(self, params, t):
    '''
    Now we assume the params space is 18
    Args:
        params(array): parameters for controller 54*1
        t(float): phase variable
    Returns:
        action(array): action acting on robot 18*1
    '''
    action = np.zeros(18)
    
    return action 

class forward_model(nn.Module):
    '''
    The forward model(mid-level policy) is a NN which is trained in a supervised manner
    '''
    def __init__(self,):
        '''
        Initialize the structure and options for model
        '''
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(state_dim + param_dim, model_hidden_dim), nn.ReLU(),
            nn.Linear(model_hidden_dim, model_hidden_dim), nn.ReLU(),
            nn.Linear(model_hidden_dim, 1))

    def forward(self,state,action):
        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)

    def curiosity_reward(self, state, next_state):
        pred_next_state = self.forward(state,action).cpu().data.numpy().flatten()
        cur_reward = np.norm(next_state - pred_next_state)
        return cur_reward


class rs_learning():
    # Random Search Learning
    def __init__(self,param_dim,state_dim):
        '''
        Initialize policy parameters and their distribution
        Initialize learning parameters
        Initialize other stuff
        '''
        #TODO: need to initialize dataset for training forward_model
        self.param_dim = param_dim
        self.policy_param = np.zeros(param_dim)
        self.policy_var = policy_var
        self.forward_model = forward_model()
        self.batch_size = batch_size
        self.sample_num = sample_num
        
        self.policy_lr = policy_lr
        self.model_lr = model_lr

        self.model_optimizer = torch.optim.Adam(self.forward_model.parameters(),lr=self.model_lr)


    def sample_params(self):
        disturb = np.random.normal(0, np.policy_var, self.sample_num)
        return self.policy_param, disturb

    def update_policy_param(self, replay_policy):
        # Basic Random Search
        self.policy_param = self.policy_param + self.lr * update_value

    def update_model(self, replay_model):
        state, action, next_state = replay_model.sample(self.batch_size)
        pred_next_state = self.forward_model(state,action)
        model_loss = F.mse_loss(pred_next_state, next_state)

        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()


    def update_all(self,replay_policy, replay_policy):
        self.update_policy_param(replay_policy)
        self.update_model(replay_model)
    
    def save_model(self):

    def save_policy(self):
    
    def save_all(self):
    
    def load_model(self):

    def load_policy(self):

    def load_all(self):
    
    