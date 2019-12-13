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

def evaluate_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = daisy_API(sim=args.sim, render=args.render, logger = False)
    env.set_control_mode(args.control_mode)
    state = env.reset()
    utils.make_dir(args.save_dir)
    save_dir = utils.make_dir(os.path.join(args.save_dir + '/trial_%s' % str(args.seed))) if args.save else None
    logger = Logger(save_dir, name = 'eval')
    logger2 = Logger(save_dir, name = 'tgt')
    total_timestep, total_latent_action = 0, 0 

    if args.sim:
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
        high_level_policy_type = args.high_level_policy_type,
        update_sample_policy = args.update_sample_policy,
        update_sample_policy_lr = args.update_sample_policy_lr,
        low_level_policy_type = args.low_level_policy_type,
        num_timestep_per_footstep = args.num_timestep_per_footstep,
        update_per_iter = args.update_per_iter,
    )
    high_level_planning.load_data(save_dir)
    
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
        # reset robot to stand 
        if args.sim:
            state = motion_library.exp_standing(env)
        # generate new target speed
        tgt_vel = np.clip(np.random.randn(2),-0.03,0.03)

        for _ in range(args.num_latent_action_per_iteration):
            # generate foot footstep position. If test, the footstep comes from optimization process
            pre_com_state = state
            if args.test:
                latent_action = high_level_planning.plan_latent_action(state)
            else:
                latent_action = high_level_planning.sample_latent_action()

            # update LLTG (target footstep position and stance & swing leg)
            low_level_TG.update_latent_action(pre_com_state,latent_action)
            
            for step in range(1, args.num_timestep_per_footstep+1):
                action = low_level_TG.get_action(state, step)
                state, total_timestep = env.step(action), total_timestep + 1

            post_com_state = state
            total_latent_action += 1
            logger.log('eval/x_vel',state['base_velocity'][0])
            logger.log('eval/y_vel',state['base_velocity'][1])
            logger.log('eval/x_pos',state['base_pos_x'][0])
            logger.log('eval/y_pos',state['base_pos_y'][0])
            logger2.log('tgt/x_vel',tgt_vel[0])
            logger2.log('tgt/y_vel',tgt_vel[1])
            logger.dump(total_latent_action)
            logger2.dump(total_latent_action)

            # Check if robot still alive
            if utils.check_data_useful(state):
                high_level_obs, high_level_delta_obs = utils.HL_obs(pre_com_state), utils.HL_delta_obs(pre_com_state, post_com_state)
                predict_high_level_delta_obs = high_level_planning.forward_model.predict(high_level_obs, latent_action)
                print(predict_high_level_delta_obs - high_level_delta_obs)



            if utils.check_robot_dead(state):
                break
 

    return 

if __name__ == "__main__":
    args = parse_args()
    if args.test:
        evaluate_model(args)
    else:
        print(" This is evaluation file, please run training file")
