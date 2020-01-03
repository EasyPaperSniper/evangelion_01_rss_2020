import torch
import numpy as np 

z_dim = 1
num_primitive = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


z_action_all = torch.as_tensor(np.random.randn((num_primitive, z_dim)),  device= device).float()