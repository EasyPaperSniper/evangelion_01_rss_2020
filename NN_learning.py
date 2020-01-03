import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class NN_tra_generator(nn.Module):
    '''
    The NN trajectory generator(low level controller) is a NN which is trained in a supervised manner
    '''
    def __init__(self, z_dim, policy_output_dim, policy_hidden_num, device):
        '''
        Initialize the structure of trajectory generator
        '''
        super().__init__()
        self.device = device
        self.trunk = nn.Sequential(
            nn.Linear(1 + z_dim, policy_hidden_num ), nn.ReLU(),
            nn.Linear(policy_hidden_num, policy_hidden_num), nn.ReLU(),
            nn.Linear(policy_hidden_num, policy_output_dim))

    def forward(self, z_action, phase):
        low_level_input = torch.cat([z_action, phase], dim=1)
        return self.trunk(low_level_input)

    def get_action(self, z_action, phase):
        latent_action = torch.FloatTensor(z_action).to(self.device)
        latent_action = latent_action.unsqueeze(0)
        phase_term = torch.FloatTensor(phase).to(self.device)
        phase_term = phase.unsqueeze(0)
        action = self.forward(latent_action, phase_term)
        return action.cpu().data.numpy().flatten()


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
        # define random z_action


    def update_model(self):
        

        # update NN
        pred_action = self.forward_model( ,phase)
        policy_loss = F.mse_loss(pred_action, expert_action)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # update each z_action


z_dim = 1
num_primitive = 5
policy_output_dim = 18
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


z_action_all = torch.as_tensor(np.random.randn((num_primitive, z_dim)),  device= device).float()


# For every expert, collect all data