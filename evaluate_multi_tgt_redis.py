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


target_position_test = [np.array([0.0, 2.0, 0.0,1]),
                            np.array([2.0, 2.0, 0.0,1]),
                            np.array([-2.0, 2.0, 0.0,1]),
                            np.array([2.0, 0.0,  0.0,1]),
                            np.array([-2.0, 0.0,  0.0,1]),
                            np.array([0.0, -2.0,  0.0,1]),
                            np.array([-2.0, -2.0,  0.0,1]),
                            np.array([2.0, -2.0,  0.0,1]),
                            ]

test_index = 1
def evaluation(args, r, high_level_planning):
    total_cost_all = np.zeros(8)
    save_dir = utils.make_dir(os.path.join(args.save_dir + '/trial_%s' % str(args.seed))) if args.save else None
    exp_variables = {
        'finish_exp': [0],
        'not_finish_one_iter': [1],
        'finish_one_step': [0],
        'update_z_action': [0],
    }
    ru.set_variables(r, exp_variables)

    for iter in range(5,args.num_iters):
        total_cost = 0
        ru.wait_for_key(r,'not_finish_one_iter', change= False)
        print('Iteration: ' , iter+1)
        input('Press any key after initialized robot')
        ru.wait_for_key(r,'finish_one_step')
        state, exp_variables = ru.get_state(r)
        target = target_position_test[iter]/2.0
        target[0]+= target[0]+ state['base_pos_x'][0]
        target[1]+= target[1]+ state['base_pos_y'][0]
        target[3] = 1
        position_tracking = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],]
        position_tracking[2*iter].append(state['base_pos_x'][0])
        position_tracking[2*iter+1].append(state['base_pos_y'][0])

        j = 0
        while True:
            pre_com_state = state

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
            
            cost = utils.easy_cost(target,pre_com_state, post_com_state)
            total_cost += cost

            position_tracking[2*iter].append(state['base_pos_x'][0])
            position_tracking[2*iter+1].append(state['base_pos_y'][0])
            print([state['base_pos_x'][0],state['base_pos_y'][0]],high_level_delta_obs[0:2],predict_state[0:2] )
            print(target)
            j+=1

            if np.linalg.norm(target[0:2] - np.array([state['base_pos_x'][0], state['base_pos_y'][0]])) < 0.25 :
                print("Reach One Goal !!!!")
                total_cost_all[iter] = total_cost
                np.save(save_dir +'/' + args.high_level_policy_type +'_multi_tgt_test_'+str(iter)+'_'+str(test_index)+'.npy', np.array(position_tracking) )
                np.save(save_dir +'/' + args.high_level_policy_type +'_multi_tgt_cost_'+str(iter)+'_'+str(test_index)+'.npy', np.array(total_cost) )
                break

            if j>30:
                print('Did not reach goal')
                break
        
        
        exp_variables['not_finish_one_iter'] = [0]
        ru.set_variables(r, exp_variables)
        input('wait for pos')
        exp_variables = ru.get_variables(r)
        state_record = exp_variables['record_pos']
        np.save(save_dir +'/' + 'state' +'_multi_tgt_test_'+str(iter)+'_'+str(test_index)+'.npy', np.array(state_record) )

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