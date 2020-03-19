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


# rollout to collect data
def collect_data(cfg,env,high_level_planning,low_level_TG, HL_replay_buffer, com_utils):
    for iter in range(cfg.num_iters):
        state = motion_library.exp_standing(env)
        low_level_TG.reset(state)
        
        for j in range(cfg.num_latent_action_per_iteration):
            # generate foot footstep position. If test, the footstep comes from optimization process
            pre_com_state = state
            if j%2==0:
                latent_action = high_level_planning.sample_latent_action()

            low_level_TG.update_latent_action(state,latent_action)
        
            for step in range(1, cfg.num_timestep_per_footstep+1):
                action = low_level_TG.get_action(state, step)
                state = env.step(action)

            post_com_state = state
            high_level_obs, high_level_delta_obs = com_utils.HL_obs(pre_com_state), com_utils.HL_delta_obs(pre_com_state, post_com_state)
            HL_replay_buffer.add(high_level_obs, latent_action, 0, high_level_delta_obs, 1)

            if utils.check_robot_dead(state):
                break
    
    HL_replay_buffer.save_buffer()


# load data buffer to train model 
def train_model(cfg, HL_replay_buffer, high_level_planning ):
    HL_replay_buffer.load_buffer()
    logger = Logger('.', name = 'train_'+ str(cfg.model_layer_num)+'_'+str(cfg.model_hidden_num))
    high_level_planning.load_mean_var('./buffer_data')
    high_level_planning.update_model(HL_replay_buffer,logger)


@hydra.main(config_path='config/LAT_2_learning_config.yaml',strict=False)
def main(cfg):
    # define env & high level planning part & low level trajectory generator & replay buffer for HLP
    # initialize logger
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = daisy_API(sim=cfg.sim, render=cfg.render, logger = False)
    env.set_control_mode(cfg.control_mode)
    state = env.reset()
    com_utils = utils.CoM_frame_MPC()
    
    if cfg.sim:
        init_state = motion_library.exp_standing(env)
        
    model_obs_dim, model_output_dim = np.size(com_utils.HL_obs(state)), np.size(com_utils.HL_delta_obs(state, state))
    
    HL_replay_buffer = utils.ReplayBuffer(model_obs_dim, cfg.z_dim, model_output_dim, device, cfg.num_iters * cfg.num_latent_action_per_iteration)

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

    # # # collect data
    collect_data(cfg,env,high_level_planning,low_level_TG, HL_replay_buffer, com_utils)

    # train model
    train_model(cfg, HL_replay_buffer, high_level_planning )


if __name__ == "__main__":
    main()
