# Make sure that roll and pitch are from world frame during control while keep yaw = 0
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch

from daisy_API import daisy_API
import daisy_hardware.motion_library as motion_library
import high_level_planning_model as HLPM
import low_level_traj_gen as LLTG
from args_all import parse_args 
import utils


def main(args):
    # define env & high level planning part & low level trajectory generator & replay buffer for HLP
    # initialize logger
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = daisy_API(sim=args.sim, render=True)
    env.set_control_mode(args.control_mode)
    state = env.reset()
    utils.make_dir(args.save_dir)
    save_dir = utils.make_dir(args.save_dir + '/trial_%s' % str(args.seed)) if args.save else None

    # TODO: use the obs generator to generate example obs and get the data structure
    if args.sim:
        init_state = motion_library.exp_standing(env)
    model_obs_dim, model_output_dim = np.size(utils.HL_obs(state)), np.size(utils.HL_delta_obs(state, state))
    
    HL_replay_buffer = utils.ReplayBuffer(model_obs_dim, args.z_dim, model_output_dim, device, args.high_level_buffer_size)

    high_level_planning = HLPM.high_level_planning(
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
    )
    
    low_level_TG = LLTG.low_level_TG(
        z_dim = args.z_dim,
        a_dim = args.a_dim,
        num_timestep_per_footstep = args.num_timestep_per_footstep,
        batch_size = args.batch_size,
        low_level_policy_type = args.low_level_policy_type,
        update_low_level_policy = args.update_low_level_policy,
        update_low_level_policy_lr = args.update_low_level_policy_lr,
        init_state = init_state,
    )
    if args.test and args.high_level_policy_type !='raibert':
        high_level_planning.load_data(save_dir) 

    for _ in range(args.num_iters):
        # reset robot to stand 
        if args.sim:
            state = motion_library.exp_standing(env)

        for _ in range(args.num_latent_action_per_iteration):
            # generate foot footstep position. If test, the footstep comes from optimization process
            pre_com_state = state
            if args.test:
                latent_action = high_level_planning.plan_latent_action(state)
            else:
                latent_action = high_level_planning.sample_latent_action()

            # TODO:
            
            
            # update LLTG (target footstep position and stance & swing leg)
            low_level_TG.update_latent_action(pre_com_state,latent_action)
            
            for step in range(1, args.num_timestep_per_footstep+1):
                action = low_level_TG.get_action(state, step)
                state = env.step(action)

            post_com_state = state
            # Check if robot still alive
            if utils.check_data_useful(state):
                high_level_obs, high_level_delta_obs = utils.HL_obs(pre_com_state), utils.HL_delta_obs(pre_com_state, post_com_state)
                HL_replay_buffer.add(high_level_obs, latent_action, 0, high_level_delta_obs, 1) # reward = 0; done = 1

            if utils.check_robot_dead(state):
                break
            
        if not args.test:
            for _ in range(args.update_per_iter):
                # model update
                high_level_planning.update_model(HL_replay_buffer)
            if args.save:
                high_level_planning.save_data(save_dir)    

    
if __name__ == "__main__":
    main(parse_args())



