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
import redis_utils as ru
from logger import Logger
from main_learning import train_model


def collect_data_client(args, r, high_level_planning, HL_replay_buffer):
    exp_variables = {
        'do_exp': [1],
        'do_one_iter': [0],
        'finish_one_step': [0],
        'updated_z_action': [1],
    }
    ru.set_variables(r, exp_variables)

    for i in range(args.num_iters):
        exp_variables['do_one_iter'] = [1]
        ru.set_variables(r, exp_variables)
        state = ru.get_state(r)

        for _ in range(args.num_latent_action_per_iteration):
            pre_com_state = ru.get_state(r)
            # target = np.clip(0.3 * np.random.randn(3),-0.4,0.4)
            target = np.zero(args.z_dim)
            # take current state and plan for next z_action and sent to daisy
            if args.test:  
                latent_action = high_level_planning.plan_latent_action(pre_com_state, target)
            else:
                latent_action = high_level_planning.sample_latent_action(target)
            
            exp_variables['z_action'] = latent_action.tolist()
            exp_variables['update_z_action'] = [1]
            ru.set_variables(r, exp_variables)

            # check if finish one step
            ru.wait_for_one_step(r)
            post_com_state = ru.get_state(r)

            if utils.check_data_useful(state):
                high_level_obs, high_level_delta_obs = utils.HL_obs(pre_com_state), utils.HL_delta_obs(pre_com_state, post_com_state)
                HL_replay_buffer.add(high_level_obs, latent_action, 0, high_level_delta_obs, 1)

            if utils.check_robot_dead(state):
                break
        
        exp_variables['do_one_iter'] = 0
        ru.set_variables(r, exp_variables)

    # experiment ends
    exp_variables['exp_variables'] = 0
    ru.set_variables(r, exp_variables)
    model_save_dir = utils.make_dir(os.path.join(args.save_dir + '/trial_%s' % str(args.seed))) if args.save else None
    HL_replay_buffer.save_buffer(model_save_dir)


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
            control_frequency = args.control_frequency
        )

    collect_data_client(args, r, high_level_planning , HL_replay_buffer)

    # train_model(args, HL_replay_buffer, high_level_planning )
   

if __name__ == "__main__":
    args = parse_args()
    main(args)