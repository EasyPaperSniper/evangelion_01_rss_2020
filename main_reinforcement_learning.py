#!/anaconda3/bin/python

# In this learning file, sample policy updates as well as data collecting

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
import utils
from logger import Logger

target_all = np.array([
    np.array([0.0, 2.0, 0.0,1]),
    # np.array([2.0, 0.0, 0.0,1]),
    np.array([-2.0, 0.0, 0.0,1]),
    # np.array([0.0, -2.0, 0.0,1]),
])

# rollout to collect data
def collect_data_train(cfg,env,high_level_planning,low_level_TG, HL_replay_buffer, com_utils):
    reward_record = []
    for iter in range(cfg.num_iters):
        state = motion_library.exp_standing(env)
        low_level_TG.reset(state)
        target = target_all[iter%np.shape(target_all)[0]]
        total_reward = 0

        for j in range(cfg.num_latent_action_per_iteration):
            # generate foot footstep position. If test, the footstep comes from optimization process
            pre_com_state = state
            high_level_obs = com_utils.HL_obs(pre_com_state,target)

            latent_action = high_level_planning.sample_latent_action(state, target, com_utils)
            low_level_TG.update_latent_action(state,latent_action)
        
            for step in range(1, cfg.num_timestep_per_footstep+1):
                action = low_level_TG.get_action(state, step)
                state = env.step(action)
            
            post_com_state = state
            high_level_obs_ = com_utils.HL_obs(post_com_state,target)
            reward = com_utils.calc_reward(post_com_state,target)
            total_reward += reward
            done = (j==(cfg.num_latent_action_per_iteration-1))

            HL_replay_buffer.add(high_level_obs, latent_action, reward, high_level_obs_, done)
        
        print('Total Reward: ', total_reward)
        reward_record.append(total_reward)
        high_level_planning.update_policy(HL_replay_buffer)
    np.save('./reward_record.npy', reward_record)



@hydra.main(config_path='config/SAC_learning_config.yaml',strict=False)
def main(cfg):
    # define env & high level planning part & low level trajectory generator & replay buffer for HLP
    # initialize logger
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = daisy_API(sim=cfg.sim, render=cfg.render, logger = False)
    env.set_control_mode(cfg.control_mode)
    state = env.reset()
    com_utils = utils.CoM_frame_RL()
    
    if cfg.sim:
        init_state = motion_library.exp_standing(env)
        
    model_obs_dim, model_output_dim = np.size(com_utils.HL_obs(state,target_all[0])) , np.size(com_utils.HL_obs(state,target_all[0]))
    
    HL_replay_buffer = utils.ReplayBuffer(model_obs_dim, cfg.z_dim, model_output_dim, device, cfg.buffer_capacity)

    high_level_planning = HLPM.high_level_planning(
        device = device,
        model_obs_dim = model_obs_dim,
        z_dim = cfg.z_dim,
        model_output_dim = model_output_dim,
        model_hidden_num = cfg.model_hidden_num,
        batch_size = cfg.batch_size,
        model_lr = cfg.model_lr,
        high_level_policy_type = cfg.high_level_policy_type,
        update_sample_policy = cfg.update_sample_policy,
        update_sample_policy_lr = cfg.update_sample_policy_lr,
        low_level_policy_type = cfg.low_level_policy_type,
        num_timestep_per_footstep = cfg.num_timestep_per_footstep,
        model_update_steps = cfg.model_update_steps,
        control_frequency=cfg.control_frequency)
    
    low_level_TG = LLTG.low_level_TG(
        device = device,
        z_dim = cfg.z_dim,
        a_dim = cfg.a_dim,
        num_timestep_per_footstep = cfg.num_timestep_per_footstep,
        batch_size = cfg.batch_size,
        low_level_policy_type = cfg.low_level_policy_type, 
        update_low_level_policy = cfg.update_low_level_policy,
        update_low_level_policy_lr = cfg.update_low_level_policy_lr,
        init_state = init_state,)

    if cfg.low_level_policy_type =='NN':
        low_level_TG.load_model('.')

    # # # collect data
    collect_data_train(cfg,env,high_level_planning,low_level_TG, HL_replay_buffer, com_utils)


if __name__ == "__main__":
    main()
    