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

# target_speed_all = [
#                 np.array([0.0, -0.0, 0.2]),
#                 np.array([0.0, -0.0, -0.2]),
#                 np.array([0.0, -0.2, 0.0]),
#                 np.array([-0.0, 0.2, 0.0]),
#                 np.array([-0.2, 0.0, 0.0]),
#                 np.array([0.2, -0.0, 0.0]),
#                 np.array([-0.1, 0.2, 0.0]),
#                 np.array([-0.2,-0.2, 0.0]),
#                 np.array([0.2, -0.2, 0.0]),
#                 np.array([0.2, 0.2, 0.0]),
#                 ]
target_speed_all = [
                np.array([0.2, 0.2, 0.0]),
                np.array([0.4, 0.4, 0.0]),
                np.array([0.2, 0.1, 0.0]),
                np.array([0.2, 0.3, 0.0]),
                np.array([0.3, 0.1, 0.0]),
                np.array([0.3, 0.2, 0.0]),
                np.array([0.4, 0.2, 0.0]),
                np.array([0.1, 0.4, 0.0]),
                np.array([0.1, 0.1, 0.0]),
                np.array([0.2, 0.2, 0.0]),
                ]


# rollout to collect data
def collect_data(cfg,env,high_level_planning,low_level_TG, HL_replay_buffer):
    for iter in range(1):
        if cfg.sim:
            state = motion_library.exp_standing(env)
            low_level_TG.reset(state)
        target_speed = target_speed_all[iter]
        target_speed = -target_speed
        
        for j in range(20):
            # generate foot footstep position. If test, the footstep comes from optimization process
            latent_action = high_level_planning.plan_latent_action(state,target_speed)
            low_level_TG.update_latent_action(state,latent_action)
        
            for step in range(50):
                action = low_level_TG.get_action(state, step+1)
                state = env.step(action)
 

@hydra.main(config_path='config/config.yaml',strict=False)
def main(cfg):
    print(cfg.pretty())

    # define env & high level planning part & low level trajectory generator & replay buffer for HLP
    # initialize logger
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = daisy_API(sim=False, render=True, logger = False)
    env.set_control_mode(cfg.control_mode)
    state = env.reset()

    init_state = motion_library.demo_standing(env, shoulder = 0.7, elbow = 0.9)
        
    model_obs_dim, model_output_dim = 2, 5
    
    HL_replay_buffer = utils.ReplayBuffer(model_obs_dim, cfg.z_dim, model_output_dim, device,cfg.num_iters * cfg.num_latent_action_per_iteration)

    high_level_planning = HLPM.high_level_planning(
        device = device,
        model_obs_dim = model_obs_dim,
        z_dim = 3,
        model_output_dim = model_output_dim,
        model_hidden_num = cfg.model_hidden_num,
        batch_size = cfg.batch_size,
        model_lr = cfg.model_lr,
        high_level_policy_type = 'raibert',
        update_sample_policy = cfg.update_sample_policy,
        update_sample_policy_lr = cfg.update_sample_policy_lr,
        low_level_policy_type = cfg.low_level_policy_type,
        num_timestep_per_footstep = 50,
        model_update_steps = cfg.model_update_steps,
        control_frequency=cfg.control_frequency
    )
    
    low_level_TG = LLTG.low_level_TG(
        device = device,
        z_dim = 3,
        a_dim = cfg.a_dim,
        num_timestep_per_footstep = 50,
        batch_size = cfg.batch_size,
        low_level_policy_type = 'IK', 
        update_low_level_policy = cfg.update_low_level_policy,
        update_low_level_policy_lr = cfg.update_low_level_policy_lr,
        init_state = init_state,
    )

    # # if args.low_level_policy_type =='NN':
    # #     low_level_TG.load_model('./save_data/trial_2')

    # # # collect data
    collect_data(cfg,env,high_level_planning,low_level_TG, HL_replay_buffer)



if __name__ == "__main__":
    main()
