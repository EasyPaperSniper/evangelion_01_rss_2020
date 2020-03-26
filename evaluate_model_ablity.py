import os
import json
import math
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import hydra

from daisy_API import daisy_API
import daisy_hardware.motion_library as motion_library
import high_level_planning_model as HLPM
import low_level_traj_gen as LLTG
from args_all import parse_args 
import utils
from logger import Logger

@hydra.main(config_path='config/LAT_4_vel_tracking_config.yaml',strict=False)
def evaluate_model(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_obs_dim, model_output_dim = 2, 5

    high_level_planning = HLPM.high_level_planning(
        device = device,
        model_obs_dim = model_obs_dim,
        z_dim = cfg.z_dim,
        model_output_dim = model_output_dim,
        model_hidden_num = cfg.model_hidden_num,
        batch_size = cfg.batch_size,
        model_lr = cfg.model_lr,
        high_level_policy_type = cfg.high_level_policy_type,
        update_sample_policy = cfg.update_sample_policy,
        update_sample_policy_lr = cfg.update_sample_policy_lr,
        low_level_policy_type = cfg.low_level_policy_type,
        num_timestep_per_footstep = cfg.num_timestep_per_footstep,
        model_update_steps = cfg.model_update_steps,
        control_frequency=cfg.control_frequency
    )
    high_level_planning.load_data('.')
    high_level_planning.load_mean_var('.'+'/buffer_data')
    mean_var = high_level_planning.all_mean_var

    sample_num = 10000
    random_action_sample = 0.4 * np.clip(np.random.randn(sample_num, cfg.z_dim),-2,2)
    random_obs_sample = 0.3 * np.clip(np.random.randn(sample_num, model_obs_dim),-2,2)

    latent_action_norm = utils.normalization(random_action_sample,mean_var[2], mean_var[3] )
    HL_obs_buffer_norm = utils.normalization(random_obs_sample, mean_var[0], mean_var[1])

    predict_delta_state_norm = high_level_planning.forward_model.predict_para(HL_obs_buffer_norm, latent_action_norm)
    predict_delta_state = utils.inverse_normalization(predict_delta_state_norm,  mean_var[4], mean_var[5])

    print(np.shape(predict_delta_state))
    np.save('./model_random_shooting.npy', predict_delta_state)



if __name__ == "__main__":
    evaluate_model()
