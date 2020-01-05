# Run on laptop/ robotdev
import os
import json
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import redis

import daisy_hardware.motion_library as motion_library
import high_level_planning_model as HLPM
import low_level_traj_gen as LLTG
from args_all import parse_args 
import utils
from logger import Logger
from main_learning import train_model

def get_state(r):
    while True:
        key_dict = r.get('exp_keys')
        if key_dict['finish_one_step']:
            key_dict['finish_one_step'] = 0
            r.set('exp_keys', key_dict)
            state = r.get('state')
            return state


def collect_data_client(args, r, high_level_planning, HL_replay_buffer):
    latent_action = np.zeros(args.z_dim)
    key_dict = {
        'do_exp': 1,
        'do_one_iter': 0,
        'finish_one_step': 0,
        'updated_z_action': 1,
    }
    latent_action_dict = {'z_action': latent_action}
    
    r.set('exp_keys', key_dict)
    r.set('z_action', latent_action_dict)


    for i in range(args.num_iters):
        key_dict['do_one_iter'] = 1
        r.set('exp_keys', key_dict)
        
        state = get_state(r)

        for _ in range(args.num_latent_action_per_iteration):
            
            pre_com_state = state

            # take current state and plan for next z_action and sent to daisy
            if args.test:  
                latent_action = high_level_planning.plan_latent_action(state, target_speed)
            else:
                latent_action = high_level_planning.sample_latent_action(target_speed)
            
            latent_action_dict['z_action'] = latent_action
            key_dict['update_z_action'] = 1
            r.set('z_action', latent_action_dict)
            r.set('exp_keys', key_dict)

            # check if finish one step
            state = get_state(r)
            post_com_state = state

            if utils.check_data_useful(state):
                high_level_obs, high_level_delta_obs = utils.HL_obs(pre_com_state), utils.HL_delta_obs(pre_com_state, post_com_state)
                HL_replay_buffer.add(high_level_obs, latent_action, 0, high_level_delta_obs, 1)

            if utils.check_robot_dead(state):
                break
        

        key_dict['do_one_iter'] = 0
        r.set('exp_keys', key_dict)  

    # experiment ends
    key_dict['do_exp'] = 0
    r.set('exp_keys', key_dict)




def main(args):
    # initial initial redis
    r = redis.Redis(host='10.10.1.2', port=6379, db=0)
    # define high level stuff
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_obs_dim, model_output_dim = 4, 6
    # model_obs_dim, model_output_dim = np.size(utils.HL_obs(state)), np.size(utils.HL_delta_obs(state, state))
    utils.make_dir(args.save_dir)
    HL_replay_buffer = utils.ReplayBuffer(model_obs_dim, args.z_dim, model_output_dim, device,                 
                args.num_iters * args.num_latent_action_per_iteration)

    high_level_planning = HLPM.high_level_planning(
            device = device,
            model_obs_dim = model_obs_dim,
            z_dim = args.z_dim,
            model_output_dim = model_output_dim,
            model_hidden_num = args.model_hidden_num,
            batch_size = args.batch_size,
            model_lr = args.model_lr,
            high_level_policy_type = args.high_level_policy_type,
            update_sample_policy = args.update_sample_policy,
            update_sample_policy_lr = args.update_sample_policy_lr,
            low_level_policy_type = args.low_level_policy_type,
            num_timestep_per_footstep = args.num_timestep_per_footstep,
            model_update_steps = args.model_update_steps,
            control_frequency= = args.control_frequency
        )

    collect_data_client(args, r, high_level_planning , HL_replay_buffer)

    train_model(args, HL_replay_buffer, high_level_planning )
   


