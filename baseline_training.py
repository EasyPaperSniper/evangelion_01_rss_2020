
import math
import random
import time
import os

import torch
import numpy as np
# here import daisy simulator
from daisy_API import daisy_API
import daisy_hardware.motion_library as motion_library
import argparse

from baseline_method import SAC
import baseline_utils as utils
import low_level_traj_gen as LLTG

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',default=0,type=int)
    parser.add_argument('--control_frequency',default=100,type=int)
    parser.add_argument('--render',default=0,type=int)
    parser.add_argument('--seed',default=3,type=int)
    parser.add_argument('--control_mode',default='position',type=str)
    parser.add_argument('--z_dim',default=3,type=int)
    parser.add_argument('--a_dim',default=18,type=int)


    parser.add_argument('--num_iters',default= 100 ,type=int)
    parser.add_argument('--num_latent_action_per_iteration',default=30,type=int)
    parser.add_argument('--num_timestep_per_footstep',default=50,type=int)
    parser.add_argument('--hidden_num',default=16,type=int)
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--capacity',default=10000,type=int)

    parser.add_argument('--actor_lr',default=1e-3,type=float)
    parser.add_argument('--critic_lr',default=1e-3,type=float)
    parser.add_argument('--initial_temperature',default=1.0,type=float)



    parser.add_argument('--low_level_policy_type',default='IK',type=str)
    parser.add_argument('--update_low_level_policy',default=0,type=int)
    parser.add_argument('--update_low_level_policy_lr',default=1e-3,type=float)
    parser.add_argument('--start_training_sample_num',default=50,type=int)
    parser.add_argument('--low_level_buffer_size',default=10000,type=int)
    parser.add_argument('--save_dir',default='./save_data',type=str)

    args = parser.parse_args()
    return args

def main(args):
    # define env & high level planning part & low level trajectory generator & replay buffer for HLP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = daisy_API(sim=args.sim, render=args.render, logger = False)
    env.set_control_mode(args.control_mode)
    state = env.reset()
    utils.make_dir(args.save_dir)
    save_dir = utils.make_dir(os.path.join(args.save_dir + '/baseline'))


    # initialize configuration used for initialize many thing/ set fake target velocity  to initialize dim 
    init_state = motion_library.exp_standing(env) # you can use any init method you like, just make sure to return a init state
    target = np.zeros(2)
    obs_dim = np.size(utils.HL_obs(state,target))

    # initialize replay buffer
    HL_replay_buffer = utils.ReplayBuffer(obs_dim, args.z_dim, device, args.capacity)
    
    # initialize SAC
    policy = SAC(device, obs_dim, args.z_dim, args.hidden_num, args.actor_lr, args.critic_lr, args.initial_temperature)
    
    
    # initialize IK solver ( need IK solver in daisyAPI repo)
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

    for iter in range(args.num_iters):
        # robot stand up to initial config and IK solver initializes the first yaw
        state = motion_library.exp_standing(env)
        low_level_TG.reset(state)

        # collecting data
        for j in range(args.num_latent_action_per_iteration):
            target = np.clip(np.random.randn(2), -0.2, 0.2)# random generate some target

            high_level_obs = utils.HL_obs(state, target)
            
            # if test then select action, if training then sample action
            if args.test:
                latent_action = policy.select_action(high_level_obs)
            else:
                latent_action = policy.sample_action(high_level_obs)
            low_level_TG.update_latent_action(state,latent_action)
        
            for step in range(1, args.num_timestep_per_footstep+1):
                action = low_level_TG.get_action(state, step)
                state = env.step(action)
            
            high_level_next_obs = utils.HL_obs(state, target)
            
            #calculate cost
            reward =  utils.calc_reward(state, target)
            
            if j== args.num_latent_action_per_iteration-1:
                done = 1
            else:
                done = 0

            HL_replay_buffer.add(high_level_obs, latent_action, reward, high_level_next_obs, done)

        # train policy/everything
        for _ in range():
            policy.update(HL_replay_buffer,
               batch_size=args.batch_size,
               discount=0.99,
               tau=0.005,
               target_entropy=None)
            policy.save(save_dir)



if __name__ == "__main__":
    args = parse_args()
    main(args)