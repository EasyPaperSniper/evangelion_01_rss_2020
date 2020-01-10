# run on Daisy
# 
import os
import json
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time

import numpy as np
import torch
import redis
from daisy_API import daisy_API

import daisy_hardware.motion_library as motion_library
import high_level_planning_model as HLPM
import low_level_traj_gen as LLTG
from args_all import parse_args 
import utils
from logger import Logger
import redis_utils as ru


def run_LLTG_IK(env, args, r, low_level_TG):
    # initialize robot
    if args.sim:
        state = motion_library.exp_standing(env)
        low_level_TG.reset(state)
    

    input(" Press anything to start")
    state = env.calc_state()
    exp_variables = get_variables(r)
    exp_variables = ru.set_state(r,state, exp_variables)

    
    while True:
        # update swing/stance leg
        low_level_TG.policy.update_swing_stance()
        
        # do IK 
        for step in range(1, args.num_timestep_per_footstep+1):
            # check if footstep update/set a key stuff
            exp_variables = get_variables(r)
            if exp_variables['updated_z_action'][0]:
                z_action = np.array(exp_variables['z_action'])
                low_level_TG.policy.update_latent_action_params(state,z_action)
                exp_variables['update_z_action'] = [0]
                set_variables(r, exp_variables)

            action = low_level_TG.get_action(state, step)
            # state = env.step(action)
            time.sleep(0.01)


        # finish one step and update to high level 
        exp_variables['finish_one_step'] = [1]
        ru.set_state(r,state, exp_variables)
        exp_variables = get_variables(r)
        if not exp_variables['do_one_iter'][0]:
            break


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    r = redis.Redis(host = 'localhost', port=6379, db = 0)

    # initialize env
    env = daisy_API(sim=args.sim, realsense = True,render=args.render, logger = False)
    env.set_control_mode(args.control_mode)
    init_state = env.calc_state()

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

    a =  input('Start client')

    while True:
        print('New Iteration ??')
        key_dict = r.get('exp_keys')
        if not key_dict['do_exp'][0]:
            break
        run_LLTG_IK(env, args, r, low_level_TG)

if __name__ == "__main__":
    args = parse_args()
    main(args)

         
    