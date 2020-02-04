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


def expert_data(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = daisy_API(sim=True, render=args.render, logger = False)
    env.set_control_mode(args.control_mode)
    state = env.reset()
    utils.make_dir(args.save_dir)
    save_dir = utils.make_dir(os.path.join(args.save_dir + '/trial_%s' % str(args.seed))) if args.save else None
    total_timestep, total_latent_action = 0, 0 


    init_state = motion_library.exp_standing(env)
    model_obs_dim, model_output_dim = np.size(utils.HL_obs(state)), np.size(utils.HL_delta_obs(state, state))

    
    high_level_planning = HLPM.high_level_planning(
        device = device,
        model_obs_dim = model_obs_dim,
        z_dim = args.z_dim,
        model_output_dim = model_output_dim,
        model_hidden_num = args.model_hidden_num,
        batch_size = args.batch_size,
        model_lr = args.model_lr,
        high_level_policy_type = 'raibert',
        update_sample_policy = args.update_sample_policy,
        update_sample_policy_lr = args.update_sample_policy_lr,
        low_level_policy_type = 'IK',
        num_timestep_per_footstep = args.num_timestep_per_footstep,
        model_update_steps = args.model_update_steps,
        control_frequency= args.control_frequency
    )
    
    low_level_TG = LLTG.low_level_TG(
        device = device,
        z_dim = args.z_dim,
        a_dim = args.a_dim,
        num_timestep_per_footstep = args.num_timestep_per_footstep,
        batch_size = args.batch_size,
        low_level_policy_type = 'IK',
        update_low_level_policy = args.update_low_level_policy,
        update_low_level_policy_lr = args.update_low_level_policy_lr,
        init_state = init_state,
    )
    if args.low_level_policy_type =='NN':
        low_level_TG.load_model('./save_data/trial_2')


    target_all = [np.array([0.1, 0.0 , 0.0 ]),
                np.array([0.0, 0.1 , 0.0 ]),
                np.array([-0.1, 0.0 , 0.0 ]),
                np.array([0.0, -0.1 , 0.0 ]),
                np.array([0.05, 0.05 , 0.05 ]),
                np.array([-0.05, -0.05 , -0.05 ]),
                np.array([0.05, -0.05 , 0.05 ]),
                np.array([0.00, -0.00 , 0.00 ]),
                np.array([0.-1, 0.1 , 0.00 ]),
                np.array([0.00, -0.2 , 0.00 ]),]

    args.num_iters = np.shape(target_all)[0]
    args.num_latent_action_per_iteration = 1
    args.num_timestep_per_footstep = 170
    expert_trajectory = np.empty(( args.num_iters*args.num_latent_action_per_iteration, args.num_timestep_per_footstep, 18))
    
    for iter in range(args.num_iters):
        # reset robot to stand 
        if args.sim: 
            state = motion_library.exp_standing(env)
            low_level_TG.reset(state)

        for j in range(args.num_latent_action_per_iteration):
            target = target_all[iter]


            latent_action = high_level_planning.plan_latent_action(state, target)
            low_level_TG.update_latent_action(state,latent_action)
            
            for step in range(1, args.num_timestep_per_footstep+1):
                action = low_level_TG.get_action(state, step)
                state, total_timestep = env.step(action), total_timestep + 1
                expert_trajectory[iter*args.num_latent_action_per_iteration + j][step-1] = action
                

    np.save('./save_data/raibert_expert_trajectory.npy',expert_trajectory)
    return 

def sinusoidal_data_collect():
    from joystick_test import expert_control, expert_control_back, expert_control_run, initial_configuration, expert_control_run_back
    init_state = initial_configuration()
    expert_trajectory = np.empty((10, 170, 18))
    for i in range(170):
        t = (i+1)/170.0 * 50
        expert_trajectory[1][i] = expert_control(t, phase=1) + init_state
        expert_trajectory[0][i] = expert_control_run_back(t,a=0.5,phase=0) + init_state
        expert_trajectory[2][i] = expert_control_back(t, phase=1.0)+ init_state
        expert_trajectory[3][i] = expert_control(t, phase=0.7) + init_state
        expert_trajectory[4][i] = expert_control_run_back(t,phase=0.3) + init_state
        expert_trajectory[5][i] = expert_control_run_back(t,phase=-0.3) + init_state
        expert_trajectory[6][i] = expert_control_run_back(t,a=0.5, phase=0.1) + init_state
        expert_trajectory[7][i] = expert_control_run_back(t, phase=-0.2) + init_state
        expert_trajectory[8][i] = expert_control(t,a = 0.5, phase=1.0) + init_state
        expert_trajectory[9][i] = expert_control(t,a=0.5, phase=1.1) + init_state
        
    np.save('./save_data/sinusoidal_expert_trajectory.npy',expert_trajectory)

def test_expert():
    from joystick_test import expert_control, expert_control_back, expert_control_run, initial_configuration, expert_control_run_back
    env = daisy_API(sim=True, render=True, logger = False)
    env.set_control_mode('position')
    state = env.reset()
    init_state = initial_configuration()
    velocity_tracking = [[],[]]
    for i in range(1000):
        action = expert_control_run_back(i,a=0.5, phase=0.01) + init_state
        state = env.step(action)
        velocity_tracking[0].append(state['base_velocity'][0])
        velocity_tracking[1].append(state['base_velocity'][1])
    np.save('./save_data/velocity_sine',velocity_tracking)


if __name__ == "__main__":
    # args = parse_args()

    # expert_data(args)

    # sinusoidal_data_collect()


    test_expert()