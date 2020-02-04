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

target_speed_all = [
                np.array([0.0,0.0,0.3]),
                np.array([0.0,0.0,-0.3]),
                np.array([0.3,0,0.0]),
                np.array([0.2,0,0]),
                np.array([-0.3,0,0]),
                # np.array([-0.2,0,0]),
                # np.array([-0.0,0.3,0]),
                # np.array([-0.0,-0.3,0]),
                # np.array([-0.2,-0.3,0]),
                # np.array([0.2,0.3,0]),
                ]
# target_speed_all = [
#                 np.array([-0.0,-0.1,-0.0]),
#                 np.array([-0.0,0.2,0]),
#                 np.array([-0.0,0.3,0]),
#                 np.array([-0.0,0.3,0]),
#                 np.array([-0.0,0.4,0]),
#                 np.array([-0.0,0.4,0]),
#                 np.array([-0.2,0.2,0]),
#                 np.array([0.2,-0.2,0]),
#                 np.array([0.0,0.5,0.0]),
#                 np.array([0.0,-0.5,-0.0]),]

# collect_action = np.empty((np.shape(target_speed_all)[0],170,18))

gait_record = np.empty((10, 6, 100))

# rollout to collect data
def collect_data(args,env,high_level_planning,low_level_TG, HL_replay_buffer):
    # for iter in range(np.shape(target_speed_all)[0]):
    for iter in range(10):
        if args.sim:
            if args.low_level_policy_type=='NN':
                state = motion_library.exp_standing(env, shoulder = 0.3, elbow = 1.3)
            else:
                state = motion_library.exp_standing(env, shoulder = 0.3, elbow = 1.3)
            low_level_TG.reset(state)
        
        target_speed = target_speed_all[iter]
        
        for j in range(50):
            
            # generate foot footstep position. If test, the footstep comes from optimization process
            pre_com_state = state
            target_speed = target_speed_all[int(j/10)]
            latent_action = high_level_planning.plan_latent_action(state,target_speed)
            low_level_TG.update_latent_action(state,latent_action)
        
            for step in range(1, 51):
                action = low_level_TG.get_action(state, step)
                state = env.step(action)
                # if j == 8 or j==9:
                #     for q in range(6):
                #         k = int(int(j - 8) * 50 + step -1  )
                #         if state['foot_pos'][q][2]<=-0.005:
                #             gait_record[iter][q][k] = 1
                #         else:
                #             gait_record[iter][q][k] = 0

                # collect_action[iter][j*85 + step-1] = action
                
            post_com_state = state

    # np.save('./save_data/trial_2/footstep_exp_33.npy',collect_action)
    # np.save('./save_data/trial_2/expert_gait_record_2.npy', gait_record)



# @hydra.main(config_path='conf/',strict=False)
def main(args):
    # define env & high level planning part & low level trajectory generator & replay buffer for HLP
    # initialize logger
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = daisy_API(sim=True, render=True, logger = False)
    env.set_control_mode(args.control_mode)
    state = env.reset()
    utils.make_dir(args.save_dir)

    if args.sim:
        if args.low_level_policy_type =='NN':
            init_state = motion_library.exp_standing(env, shoulder = 0.3, elbow = 1.3)
        else:
            init_state = motion_library.exp_standing(env, shoulder = 0.3, elbow = 1.3)
        
    model_obs_dim, model_output_dim = np.size(utils.HL_obs(state)), np.size(utils.HL_delta_obs(state, state))
    
    HL_replay_buffer = utils.ReplayBuffer(model_obs_dim, args.z_dim, model_output_dim, device,args.num_iters * args.num_latent_action_per_iteration)

    high_level_planning = HLPM.high_level_planning(
        device = device,
        model_obs_dim = model_obs_dim,
        z_dim = 3,
        model_output_dim = model_output_dim,
        model_hidden_num = args.model_hidden_num,
        batch_size = args.batch_size,
        model_lr = args.model_lr,
        high_level_policy_type = 'raibert',
        update_sample_policy = args.update_sample_policy,
        update_sample_policy_lr = args.update_sample_policy_lr,
        low_level_policy_type = args.low_level_policy_type,
        num_timestep_per_footstep = 50,
        model_update_steps = args.model_update_steps,
        control_frequency=args.control_frequency
    )
    
    low_level_TG = LLTG.low_level_TG(
        device = device,
        z_dim = 3,
        a_dim = args.a_dim,
        num_timestep_per_footstep = 50,
        batch_size = args.batch_size,
        low_level_policy_type = 'IK', 
        update_low_level_policy = args.update_low_level_policy,
        update_low_level_policy_lr = args.update_low_level_policy_lr,
        init_state = init_state,
    )

    # if args.low_level_policy_type =='NN':
    #     low_level_TG.load_model('./save_data/trial_2')

    # # # collect data
    collect_data(args,env,high_level_planning,low_level_TG, HL_replay_buffer)



if __name__ == "__main__":
    args = parse_args()
    main(args)
