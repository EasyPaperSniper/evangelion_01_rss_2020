import os
import json
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def env_step(state, action):
    reward = 0
    return reward

class low_level_policy(nn.Module):
    def __init__(self, model_obs_dim , z_dim , model_output_dim, model_hidden_num, device):
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

    

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    z_dim = 3
    z_action = np.zeros(z_dim)
    LLP = low_level_policy(
        model_obs_dim = 1,
        z_dim = z_dim,
        model_output_dim = 3,
        model_hidden_num = 32,
        device = device
    )

    step = 1
    for _ in range(1000):
        state = np.array([(step%50)/50.0])
        action = LLP.predict( model_obs = state,latent_action = z_action)
        reward =  env_step(state, action)
        step+=1
    

if __name__ == "__main__":
    main()


