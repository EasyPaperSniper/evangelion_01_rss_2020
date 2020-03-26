import os
import json
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

@hydra.main(config_path='config/LAT_2_vel_tracking_config.yaml',strict=False)
def evaluate_model(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = daisy_API(sim=cfg.sim, render=cfg.render, logger = False)
    env.set_control_mode(cfg.control_mode)
    state = env.reset()
    com_utils = utils.CoM_frame_MPC()
    
    if cfg.sim:
        init_state = motion_library.exp_standing(env)
        
    model_obs_dim, model_output_dim = np.size(com_utils.HL_obs(state)), np.size(com_utils.HL_delta_obs(state, state))

    high_level_planning = HLPM.high_level_planning(
        device = device,
        model_obs_dim = model_obs_dim,
        z_dim = cfg.z_dim,
        model_output_dim = model_output_dim,
        model_hidden_num = cfg.model_hidden_num,
        model_layer_num = cfg.model_layer_num,
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
    
    low_level_TG = LLTG.low_level_TG(
        device = device,
        z_dim = cfg.z_dim,
        a_dim = cfg.a_dim,
        num_timestep_per_footstep = cfg.num_timestep_per_footstep,
        batch_size = cfg.batch_size,
        low_level_policy_type = cfg.low_level_policy_type, 
        update_low_level_policy = cfg.update_low_level_policy,
        update_low_level_policy_lr = cfg.update_low_level_policy_lr,
        init_state = init_state,
    )

    if cfg.low_level_policy_type =='NN':
        low_level_TG.load_model('.')

    prediction_evaluation = np.empty((2* model_output_dim,cfg.num_latent_action_per_iteration ))
    velocity_tracking = np.empty((cfg.num_iters,6,cfg.num_latent_action_per_iteration))

    
    target_velocity_test = [ 
                        np.array([0.0, 0.2, 0.0,1]),
                        np.array([0.0, 0.2, 0.0,1]),
                        np.array([-0.0, 0.2, 0.0,1]),
                        np.array([-0.2, 0.0,0.0,1]),
                        np.array([-0.2, 0.0, 0.0,1]),
                        np.array([-0.2, 0.0,0.0,1]),
                        np.array([0.0, -0.2,0.0,1]),                   
                        np.array([0.0, -0.2,0.0,1]),
                        np.array([0.0, -0.2, 0.0,1]),
                        np.array([0.0, -0.2, 0.0,1]),
                        np.array([0.2, -0.0, 0.0,1]),
                        np.array([0.2, -0.0, 0.0,1]),
                        np.array([0.2, -0.0, 0.0,1]),
                        np.array([0.2, -0.0, 0.0,1]),
                        np.array([0.2, -0.0,0.0,1]),
                      ]
    
    velocity_tracking = np.empty((cfg.num_iters, 6,cfg.num_latent_action_per_iteration))
    for iter in range(cfg.num_iters):
        prediction_evaluation = np.empty((2* model_output_dim,cfg.num_latent_action_per_iteration ))
        
        total_cost = 0
        init_state = motion_library.exp_standing(env)

        for j in range(cfg.num_latent_action_per_iteration):
            if not j%5:
                target = target_velocity_test[int((j+1)/5)]

            pre_com_state = state
            latent_action = high_level_planning.plan_latent_action(state, target, com_utils, cfg.mpc_sample_num, cfg.mpc_horizon)
            low_level_TG.update_latent_action(state,latent_action)
            
            for step in range(1, cfg.num_timestep_per_footstep+1):
                action = low_level_TG.get_action(state, step)
                state = env.step(action)

            post_com_state = state
            
            # Check if robot still alive
            high_level_obs , high_level_delta_obs = com_utils.HL_obs(pre_com_state), com_utils.HL_delta_obs(pre_com_state, post_com_state)
            predict_delta_state = high_level_planning.model_predict(high_level_obs, latent_action)
            predict_state_world = com_utils.predict2world(pre_com_state, predict_delta_state)

            # collect data
            velocity_tracking[iter][0][j] = target[0]
            velocity_tracking[iter][1][j] = target[1]
            velocity_tracking[iter][2][j] = predict_state_world[3]
            velocity_tracking[iter][3][j] = predict_state_world[4]
            velocity_tracking[iter][4][j] = post_com_state['base_velocity'][0]
            velocity_tracking[iter][5][j] = post_com_state['base_velocity'][1]

      
    np.save('./' + cfg.high_level_policy_type + '_velocity_tracking_fin.npy', velocity_tracking) 
    return 

if __name__ == "__main__":
    evaluate_model()
