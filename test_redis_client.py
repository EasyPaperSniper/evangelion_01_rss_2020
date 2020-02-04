# Run on laptop/ robotdev
import os
import json
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import datetime

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
# from main_learning import train_model

latent_all = [[ 0.2295,  0.5950],
        [ 0.1303,  0.0111],
        [-0.0920,  0.3827],
        [ 0.2068,  0.1836],
        [ 0.2663, -0.0928],
        [ 0.2584,  0.1057],
        [ 0.2051, -0.3227],
        [ 0.3041,  0.1780],
        [ 0.5115,  0.1338],
        [ 0.0178,  0.1417],
        [-0.3598, -0.3446],
        [-0.2903, -0.1586],
        [-0.3960, -0.3700],
        [-0.1368, -0.1423],
        [-0.5574, -0.2970],
        [-0.3518,  0.2172],
        [-0.1867, -0.6132],
        [-0.1863, -0.2268],
        [ 0.5526, -0.3470],
        [-0.2774,  0.1772]]

def train_model(args, HL_replay_buffer, high_level_planning ):
    model_save_dir = utils.make_dir(os.path.join(args.save_dir + '/trial_%s' % str(args.seed))) if args.save else None
    logger = Logger(model_save_dir, name = 'train')
    # HL_replay_buffer.load_buffer(model_save_dir )
    high_level_planning.load_mean_var(model_save_dir  + '/buffer_data')
    
    high_level_planning.update_model(HL_replay_buffer,logger)
    high_level_planning.save_data(model_save_dir)  

def collect_data_client(args, r, high_level_planning, HL_replay_buffer):
    exp_variables = {
        'finish_exp': [0],
        'not_finish_one_iter': [1],
        'finish_one_step': [0],
        'update_z_action': [0],
    }
    ru.set_variables(r, exp_variables)
    for i in range(17 ,args.num_iters):
        
        ru.wait_for_key(r,'not_finish_one_iter', change= False)
        print('Iteration: ' , i+1)
        input('Press any key after initialized robot')
        ru.wait_for_key(r,'finish_one_step')
        state, exp_variables = ru.get_state(r)

        for k in range(args.num_latent_action_per_iteration):
            pre_com_state = state
            
            q = k % 5
            if q == 0:
                target = np.clip(0.2 * np.random.randn(3),-0.25,0.25)
                latent_action = np.array(latent_all[i]) + 0.1 * np.random.randn(args.z_dim)
                # latent_action = high_level_planning.sample_latent_action(target)
            if  q==3 or q==4:
                target = np.zeros(3)
                latent_action = np.array([ 0.2068,  0.1836],) + 0.1 * np.random.randn(args.z_dim)
            
            # take current state and plan for next z_action and sent to daisy
            t_start = datetime.datetime.now()
            # if args.test:  
            #     latent_action = high_level_planning.plan_latent_action(pre_com_state, target)
            # else:
            #     latent_action = high_level_planning.sample_latent_action(target)
            
            

            t_end = datetime.datetime.now()
            
            exp_variables['z_action'] = latent_action.tolist()
            exp_variables['update_z_action'] = [1]
            ru.set_variables(r, exp_variables)

            # check if finish one step
            ru.wait_for_key(r, 'finish_one_step')
            state, exp_variables = ru.get_state(r)
            post_com_state = state 

            if utils.check_data_useful(post_com_state):
                high_level_obs, high_level_delta_obs = utils.HL_obs(pre_com_state), utils.HL_delta_obs(pre_com_state, post_com_state)
                HL_replay_buffer.add(high_level_obs, latent_action, 0, high_level_delta_obs, 1)

            if utils.check_robot_dead(post_com_state):
                break

        exp_variables['not_finish_one_iter'] = [0]
        ru.set_variables(r, exp_variables)

        if (i+1)%1 ==0:
            model_save_dir = utils.make_dir(os.path.join(args.save_dir + '/trial_%s' % str(args.seed))) if args.save else None
            HL_replay_buffer.save_buffer(model_save_dir)
    
    # experiment ends
    exp_variables['finish_exp'] = [1]
    ru.set_variables(r, exp_variables)
    

def main(args):
    r = redis.Redis(host='10.10.1.2', port=6379, db=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_obs_dim, model_output_dim = 4, 6
    # model_obs_dim, model_output_dim = np.size(utils.HL_obs(state)), np.size(utils.HL_delta_obs(state, state))
    utils.make_dir(args.save_dir)
    HL_replay_buffer = utils.ReplayBuffer(model_obs_dim, args.z_dim, model_output_dim, device,                 
                args.num_iters * args.num_latent_action_per_iteration)
    HL_replay_buffer.load_buffer('./save_data/trial_4')
    # HL_replay_buffer.idx = 1415
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
    # collect_data_client(args, r, high_level_planning , HL_replay_buffer)

    train_model(args, HL_replay_buffer, high_level_planning )
   

if __name__ == "__main__":
    args = parse_args()
    main(args)