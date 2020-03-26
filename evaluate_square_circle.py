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


@hydra.main(config_path='config/LAT_2_traj_tracking_config.yaml',strict=False)
def evaluate_model(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = daisy_API(sim=cfg.sim, render=False, logger = False)
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

    

    square_circle_test = []
    total_num = 6
    for i in range(1, total_num+1):
        theta = i *  math.pi / float(total_num)
        square_circle_test.append(np.array([1-math.cos(theta), 1.5*math.sin(theta), 0,1]))

    for i in range(1, total_num+1):
        theta = i *  math.pi / float(total_num)
        square_circle_test.append(np.array([3-math.cos(theta), -1.5*math.sin(theta), 0,1]))

    square_cost = []

    for iter in range(cfg.num_iters):
        position_tracking = [[], [], []]
        # reset robot to stand 
        state = motion_library.exp_standing(env)
        low_level_TG.reset(state)
        position_tracking[0].append(state['base_pos_x'][0])
        position_tracking[1].append(state['base_pos_y'][0])
        position_tracking[2].append(state['base_ori_euler'][2])
        
        total_latent_action = 0
        total_cost = 0
        target_index = 0
        while True:
            target =  square_circle_test[target_index]

            pre_com_state = state
            latent_action = high_level_planning.plan_latent_action(state, target, com_utils, cfg.mpc_sample_num, cfg.mpc_horizon)
            low_level_TG.update_latent_action(state,latent_action)
            
            for step in range(1, cfg.num_timestep_per_footstep+1):
                action = low_level_TG.get_action(state, step)
                state = env.step(action)
                # collect data
                position_tracking[0].append(state['base_pos_x'][0])
                position_tracking[1].append(state['base_pos_y'][0])
                position_tracking[2].append(state['base_ori_euler'][2])
                if np.linalg.norm(target[0:2] - np.array([state['base_pos_x'][0], state['base_pos_y'][0]]))  < 0.15 :
                    print("Reach Goal %s!!!!" %str(target_index))
                    target_index +=1
                    if target_index >= np.shape(square_circle_test)[0]:
                        break
                    target =  square_circle_test[target_index]
                    

            post_com_state = state
            
            total_latent_action += 1

            if target_index >= np.shape(square_circle_test)[0]:
                np.save('./square_circle_test_'+str(iter) +'.npy', np.array(position_tracking) )
                break
            
            

            if total_latent_action>200:
                print('Did not reach goal')
                break

    return 

if __name__ == "__main__":
    evaluate_model()