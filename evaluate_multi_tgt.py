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
    env = daisy_API(sim=args.sim, render=False, logger = False)
    env.set_control_mode(args.control_mode)
    state = env.reset()
    utils.make_dir(args.save_dir)
    save_dir = utils.make_dir(os.path.join(args.save_dir + '/trial_%s' % str(args.seed))) if args.save else None
    total_timestep, total_latent_action = 0, 0 

    if args.sim:
        if args.low_level_policy_type =='NN':
            init_state = motion_library.exp_standing(env, shoulder = 0.3, elbow = 1.3)
        else:
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
        control_frequency= args.control_frequency
    )
    high_level_planning.load_data(save_dir)
    high_level_planning.load_mean_var(save_dir+'/buffer_data')
    
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
        low_level_TG.load_model('./save_data/trial_2')
    position_tracking_all = np.empty((5,16,25))
    
    target_velocity_test = [np.array([0.0, 0.1, 0.0]),
                            np.array([0.05, 0.15, 0.0]),
                            np.array([0.05, 0.20, 0.0]),
                            np.array([-0.05, 0.0, 0.0]),
                            np.array([0.0, -0.1, 0.0]),
                            np.array([0.0, -0.2, 0.0]),                   
                            np.array([-0.05, -0.15, 0.0]),
                            np.array([-0.1, -0.1, 0.0]),
                            np.array([-0.0, 0.05, 0.0]),
                            np.array([0.05, 0.15, 0.0]),]
    target_position_test = [np.array([0.0, 2.0, 0.0,1]),
                            np.array([2.0, 2.0, 0.0,1]),
                            np.array([-2.0, 2.0, 0.0,1]),
                            np.array([2.0, 0.0, 0.0,1]),
                            np.array([-2.0, 0.0, 0.0,1]),
                            np.array([0.0, -2.0, 0.0,1]),
                            np.array([-2.0, -2.0, 0.0,1]),
                            np.array([2.0, -2.0, 0.0,1]),]

    total_cost_all = np.empty((5, 8))
    total_step_save = np.empty((5, 8))
    for q in range(5):
        for iter in range(0,args.num_iters):
            total_cost = 0
            # reset robot to stand 
            if args.sim: 
                if args.low_level_policy_type=='NN':
                    state = motion_library.exp_standing(env, shoulder = 0.3, elbow = 1.3)
                else:
                    state = motion_library.exp_standing(env)
                low_level_TG.reset(state)
                position_tracking_all[q][2*iter][0] = state['base_pos_x'][0]
                position_tracking_all[q][2*iter+1][0] = state['base_pos_y'][0]

            for j in range(1,25):
                target = target_position_test[iter]

                pre_com_state = state
                latent_action = high_level_planning.plan_latent_action(state, target)
                low_level_TG.update_latent_action(state,latent_action)
                
                for step in range(1, args.num_timestep_per_footstep+1):
                    action = low_level_TG.get_action(state, step)
                    state, total_timestep = env.step(action), total_timestep + 1

                
                # collect data
                position_tracking_all[q][2*iter][j] = state['base_pos_x'][0]
                position_tracking_all[q][2*iter+1][j] = state['base_pos_y'][0]

                post_com_state = state
                cost = utils.easy_cost(target,pre_com_state, post_com_state)
                total_cost += cost

                # Check if robot still alive
                if utils.check_data_useful(state):
                    high_level_obs , high_level_delta_obs = utils.HL_obs(pre_com_state), utils.HL_delta_obs(pre_com_state, post_com_state)
                    predict_state = high_level_planning.model_predict(high_level_obs, latent_action)

                
        
                total_latent_action += 1


                if utils.check_robot_dead(state):
                    break
                
                if np.linalg.norm(target[0:2] - np.array([state['base_pos_x'][0], state['base_pos_y'][0]])) < 0.2 :
                    total_cost_all[q][iter] = total_cost
                    total_step_save[q][iter] = j
                    for k in range(j,25):
                        position_tracking_all[q][2*iter][k] = state['base_pos_x'][0]
                        position_tracking_all[q][2*iter+1][k] = state['base_pos_y'][0]

                    print("ReachGoal" + str(iter)+ "!!!!")
                    break

    # print(total_cost_all)
    np.save(save_dir + '/multi_tgt_steps.npy', total_step_save )
    # np.save(save_dir + '/position_tracking_not_fixed.npy', position_tracking_all )
    # np.save(save_dir + '/position_tracking_total_cost.npy', total_cost_all )
    
    return 

if __name__ == "__main__":
    args = parse_args()
    if args.test:
        evaluate_model(args)
    else:
        print(" This is evaluation file, please run training file")