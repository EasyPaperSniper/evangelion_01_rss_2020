import os
import json
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch

from daisy_API import daisy_API
import daisy_hardware.motion_library as motion_library
import high_level_planning_model as HLPM
import low_level_traj_gen as LLTG
from args_all import parse_args 
import utils
from logger import Logger


# rollout to collect data
def collect_data(args,env,high_level_planning,low_level_TG, HL_replay_buffer):
    for iter in range(args.num_iters):
        if args.sim:
            state = motion_library.exp_standing(env)
            low_level_TG.reset(state)
        
        for _ in range(args.num_latent_action_per_iteration):
            target_speed = np.clip(0.3 * np.random.randn(3),-0.4,0.4)
            # generate foot footstep position. If test, the footstep comes from optimization process
            pre_com_state = state

            latent_action = high_level_planning.sample_latent_action(target_speed)

            low_level_TG.update_latent_action(state,latent_action)
        
            for step in range(1, args.num_timestep_per_footstep+1):
                action = low_level_TG.get_action(state, step)
                state = env.step(action)

            post_com_state = state
            # Check if robot still alive
            if utils.check_data_useful(state):
                high_level_obs, high_level_delta_obs = utils.HL_obs(pre_com_state), utils.HL_delta_obs(pre_com_state, post_com_state)
                HL_replay_buffer.add(high_level_obs, latent_action, 0, high_level_delta_obs, 1)

            if utils.check_robot_dead(state):
                break
    
    model_save_dir = utils.make_dir(os.path.join(args.save_dir + '/trial_%s' % str(args.seed))) if args.save else None
    HL_replay_buffer.save_buffer(model_save_dir)


# load data buffer to train model 
def train_model(args, HL_replay_buffer, high_level_planning ):
    model_save_dir = utils.make_dir(os.path.join(args.save_dir + '/trial_%s' % str(args.seed))) if args.save else None
    logger = Logger(model_save_dir, name = 'train')
    HL_replay_buffer.load_buffer(model_save_dir )
    high_level_planning.load_mean_var(model_save_dir  + '/buffer_data')
    
    high_level_planning.update_model(HL_replay_buffer,logger)
    high_level_planning.save_data(model_save_dir)  



def main(args):
    # define env & high level planning part & low level trajectory generator & replay buffer for HLP
    # initialize logger
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = daisy_API(sim=args.sim, render=args.render, logger = False)
    env.set_control_mode(args.control_mode)
    state = env.reset()
    utils.make_dir(args.save_dir)

    if args.sim:
        if args.low_level_policy_type =='NN':
            init_state = motion_library.exp_standing(env, shoulder = 0.3, elbow = 1.3)
        else:
            init_state = motion_library.exp_standing(env)
        
    model_obs_dim, model_output_dim = np.size(utils.HL_obs(state)), np.size(utils.HL_delta_obs(state, state))
    
    HL_replay_buffer = utils.ReplayBuffer(model_obs_dim, args.z_dim, model_output_dim, device,args.num_iters * args.num_latent_action_per_iteration)

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
        control_frequency=args.control_frequency
    )
    
    low_level_TG = LLTG.low_level_TG(
        device = device,
        z_dim = args.z_dim,
        a_dim = args.a_dim,
        num_timestep_per_footstep = args.num_timestep_per_footstep,
        batch_size = args.batch_size,
        low_level_policy_type = args.low_level_policy_type, 
        update_low_level_policy = args.update_low_level_policy,
        update_low_level_policy_lr = args.update_low_level_policy_lr,
        init_state = init_state,
    )

    if args.low_level_policy_type =='NN':
        low_level_TG.load_model('./save_data/trial_1')

    # # # collect data
    collect_data(args,env,high_level_planning,low_level_TG, HL_replay_buffer)

    # train model
    train_model(args, HL_replay_buffer, high_level_planning )


if __name__ == "__main__":
    args = parse_args()
    if not args.test:
        main(args)
    else:
        print(" This is training file, please run evaluation file")

