import torch
import redis
import math
import os
import json
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import daisy_hardware.motion_library as motion_library

import high_level_planning_model as HLPM
import low_level_traj_gen as LLTG
from args_all import parse_args 
import utils
import redis_utils as ru
from logger import Logger
from main_learning import train_model


target_velocity_test = [np.array([0.0, 0.15, 0.0,1]),
                        np.array([0.0, 0.15, 0.0,1]),
                        np.array([-0.0, 0.15, 0.0,1]),
                        np.array([-0.15, 0.0,0.0,1]),
                        np.array([-0.15, 0.0, 0.0,1]),
                        np.array([-0.15, 0.0,0.0,1]),
                        np.array([0.0, -0.15,0.0,1]),                   
                        np.array([0.0, -0.15,0.0,1]),
                        np.array([0.00, -0.15, 0.0,1]),
                        np.array([0.0, -0.15, 0.0,1]),
                        np.array([0.15, -0.0, 0.0,1]),
                        np.array([0.15, -0.0, 0.0,1]),
                        np.array([0.15, -0.0, 0.0,1]),
                        np.array([0.15, -0.0, 0.0,1]),
                        np.array([0.15, -0.0,0.0,1]),]


def evaluation(args, r, high_level_planning):
    velocity_tracking = np.empty((6,args.num_iters,args.num_latent_action_per_iteration))
    save_dir = utils.make_dir(os.path.join(args.save_dir + '/trial_%s' % str(args.seed))) if args.save else None
    exp_variables = {
        'finish_exp': [0],
        'not_finish_one_iter': [1],
        'finish_one_step': [0],
        'update_z_action': [0],
    }
    ru.set_variables(r, exp_variables)

    for i in range(0,args.num_iters):
        
        ru.wait_for_key(r,'not_finish_one_iter', change= False)
        print('Iteration: ' , i+1)
        input('Press any key after initialized robot')
        ru.wait_for_key(r,'finish_one_step')
        state, exp_variables = ru.get_state(r)

        for j in range(args.num_latent_action_per_iteration):
            pre_com_state = state
            if not j%5:
                target = target_velocity_test[int((j+1)/5)]/1.5*2

                       
            # take current state and plan for next z_action and sent to daisy
            latent_action = high_level_planning.plan_latent_action(pre_com_state, target)
 
            exp_variables['z_action'] = latent_action.tolist()
            exp_variables['update_z_action'] = [1]
            ru.set_variables(r, exp_variables)

            # check if finish one step
            ru.wait_for_key(r, 'finish_one_step')
            state, exp_variables = ru.get_state(r)
            post_com_state = state 

            # if utils.check_data_useful(post_com_state):
            high_level_obs, high_level_delta_obs = utils.HL_obs(pre_com_state), utils.HL_delta_obs(pre_com_state, post_com_state)
            predict_state = high_level_planning.model_predict(high_level_obs, latent_action)


            velocity_tracking[0][i][j] = target[0]
            velocity_tracking[1][i][j] = target[1]
            velocity_tracking[2][i][j] = predict_state[4]
            velocity_tracking[3][i][j] = predict_state[5]
            velocity_tracking[4][i][j] = high_level_delta_obs[4]
            velocity_tracking[5][i][j] = high_level_delta_obs[5]
        
        np.save(save_dir +'/' + args.high_level_policy_type +'_velocity_tracking_test.npy', velocity_tracking) 
        exp_variables['not_finish_one_iter'] = [0]
        ru.set_variables(r, exp_variables)


    
    exp_variables = ru.get_variables(r)
    velocity_record = exp_variables['record_vel']
    np.save(save_dir +'/' + args.high_level_policy_type +'_velocity_record.npy', np.array(velocity_record))
    # experiment ends
    exp_variables['finish_exp'] = [1]
    ru.set_variables(r, exp_variables)
    


def main(args):
    r = redis.Redis(host='10.10.1.2', port=6379, db=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_obs_dim, model_output_dim = 4, 6
    # model_obs_dim, model_output_dim = np.size(utils.HL_obs(state)), np.size(utils.HL_delta_obs(state, state))
    utils.make_dir(args.save_dir)
    save_dir = utils.make_dir(os.path.join(args.save_dir + '/trial_%s' % str(args.seed))) if args.save else None

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
            control_frequency = args.control_frequency
        )
    high_level_planning.load_data(save_dir)
    high_level_planning.load_mean_var(save_dir+'/buffer_data')

    evaluation(args, r, high_level_planning)

   

if __name__ == "__main__":
    args = parse_args()
    main(args)