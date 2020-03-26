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


@hydra.main(config_path='config/LAT_2_multi_tgt_config.yaml',strict=False)
def evaluate_model(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = daisy_API(sim=cfg.sim, render=cfg.render, logger = False)
    env.set_control_mode(cfg.control_mode)
    state = env.reset()
    com_utils = utils.CoM_frame_MPC()
    if cfg.high_level_policy_type== 'SAC':
        com_utils = utils.CoM_frame_RL()
    
    if cfg.sim:
        init_state = motion_library.exp_standing(env)

    if cfg.high_level_policy_type== 'SAC':
        model_obs_dim, model_output_dim = np.size(com_utils.HL_obs(state,np.zeros(3))) , np.size(com_utils.HL_obs(state,np.zeros(3)))
    else:   
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
    if cfg.high_level_policy_type!= 'SAC': 
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
    
    
    target_position_test = [np.array([0.0, 2.0, 0.0,1]),
                            np.array([2.0, 2.0, 0.0,1]),
                            np.array([-2.0, 2.0, 0.0,1]),
                            np.array([2.0, 0.0, 0.0,1]), 
                            np.array([-2.0, 0.0, 0.0,1]),
                            np.array([0.0, -2.0, 0.0,1]),
                            np.array([-2.0, -2.0, 0.0,1]),
                            np.array([2.0, -2.0, 0.0,1]),
                            ]
    
    
    for q in range(cfg.num_iters):
        position_tracking = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],]
        for target_index in range(8):

            # reset robot to stand 
            state = motion_library.exp_standing(env)
            low_level_TG.reset(state)
            
            position_tracking[2*target_index].append(state['base_pos_x'][0])
            position_tracking[2*target_index+1].append(state['base_pos_y'][0])
            total_latent_action = 0
            target = target_position_test[target_index] 
            target[0] = target[0] + state['base_pos_x'][0]
            target[1] = target[1] + state['base_pos_y'][0]
            
            while True:
                pre_com_state = state
                latent_action = high_level_planning.plan_latent_action(state, target, com_utils, cfg.mpc_sample_num, cfg.mpc_horizon)
                low_level_TG.update_latent_action(state,latent_action)
                
                for step in range(1, cfg.num_timestep_per_footstep+1):
                    action = low_level_TG.get_action(state, step)
                    state = env.step(action)
                    position_tracking[2*target_index].append(state['base_pos_x'][0])
                    position_tracking[2*target_index+1].append(state['base_pos_y'][0])
                    if np.linalg.norm(target[0:2] - np.array([state['base_pos_x'][0], state['base_pos_y'][0]])) < 0.2 :
                        print("ReachGoal" + str(target_index)+ "!!!!")
                        break

                # collect data
                post_com_state = state                
                total_latent_action += 1

                if np.linalg.norm(target[0:2] - np.array([state['base_pos_x'][0], state['base_pos_y'][0]])) < 0.2 :
                    break
                if total_latent_action>30:
                    break
        
        for i in range(8):
            position_tracking[i] = np.array(position_tracking[i])
        np.save('./'+ cfg.high_level_policy_type +'_multi_tgt_test_'+str(q)+'.npy', np.array(position_tracking) )

    
    return 

if __name__ == "__main__":
    evaluate_model()