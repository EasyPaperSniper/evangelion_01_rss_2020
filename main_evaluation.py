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
        model_update_steps = args.model_update_steps,
    )
    high_level_planning.load_data(save_dir)
    high_level_planning.load_mean_var(args.save_dir + '/buffer_data')
    
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
        high_level_planning.policy.target_speed = np.clip(np.array([0, np.random.randn(1)]),-0.1,0.1)

        for _ in range(args.num_latent_action_per_iteration):
            # generate foot footstep position. If test, the footstep comes from optimization process
            pre_com_state = state
            if args.test:
                latent_action = high_level_planning.plan_latent_action(state)
            else:
                latent_action = high_level_planning.sample_latent_action()
            record_latent_action = np.copy(latent_action)
                    
            # update LLTG (target footstep position and stance & swing leg)

            
            for i in range(2): # half full cycle
                intermid_state = state
                if i ==1 and args.high_level_policy_type =='raibert':
                    _= high_level_planning.plan_latent_action(state)
                
                latent_action[0:args.z_dim-1] = (-1)** i  * latent_action[0:args.z_dim-1]
                # update LLTG (target footstep position and stance & swing leg)
                low_level_TG.update_latent_action(intermid_state,latent_action)
                
                for step in range(1, args.num_timestep_per_footstep+1):
                    action = low_level_TG.get_action(state, step)
                    state, total_timestep = env.step(action), total_timestep+1

            post_com_state = state
            total_latent_action += 1

            # Check if robot still alive
            if utils.check_data_useful(state):
                high_level_obs, high_level_delta_obs = utils.HL_obs(pre_com_state), utils.HL_delta_obs(pre_com_state, post_com_state)
                predict_state = high_level_planning.model_predict(high_level_obs, record_latent_action)

            # collect data
            for term in range(model_output_dim):
                logger.log('eval/term_' + str(term), predict_state[term])
                logger2.log('tgt/term_'+ str(term), high_level_delta_obs[term])
            logger.dump(total_latent_action)
            logger2.dump(total_latent_action)


            if utils.check_robot_dead(state):
                break

    return 

if __name__ == "__main__":
    args = parse_args()
    if args.test:
        evaluate_model(args)
    else:
        print(" This is evaluation file, please run training file")
