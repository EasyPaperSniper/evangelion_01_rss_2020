# run on Daisy
# 
import os
import json
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import datetime
from statistics import mean

import numpy as np
import torch
import redis
from daisy_API import daisy_API
import matplotlib.pyplot as plt

import daisy_hardware.motion_library as motion_library
import high_level_planning_model as HLPM
import low_level_traj_gen as LLTG
from args_all import parse_args 
import utils
from logger import Logger
import redis_utils as ru

SHOULDER = 0.7
ELBOW = 0.9
STACK_LENGTH = 100

def run_LLTG_IK(env, args, r):
    # initialize robot
    a = input('Select way to initialize the robot')
    if str(a) == '1': #start from ground
        state = motion_library.demo_standing(env, shoulder = SHOULDER, elbow = ELBOW)
    else:
        state = motion_library.hold_position(env, shoulder = SHOULDER, elbow = ELBOW)
    print('Robot initialized, press key on client!')

    velocity_stack = [[],[]]
    velocity_record = [[],[]]
    postion_record = [[],[],[]]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    low_level_TG = LLTG.low_level_TG(
            device = device,
            z_dim = args.z_dim,
            a_dim = args.a_dim,
            num_timestep_per_footstep = args.num_timestep_per_footstep,
            batch_size = args.batch_size,
            low_level_policy_type = args.low_level_policy_type,
            update_low_level_policy = args.update_low_level_policy,
            update_low_level_policy_lr = args.update_low_level_policy_lr,
            init_state = state,
        )
    des_height = state['base_pos_z'][0]

    exp_variables = ru.get_variables(r)
    exp_variables['finish_one_step'] = [1]
    velocity_stack[0].append(0)
    velocity_stack[1].append(0)
    exp_variables = ru.set_state(r,state, exp_variables)
    state = motion_library.hold_position(env, shoulder = SHOULDER, elbow = ELBOW)
    print('Start operation!')

    while True:
        # update swing/stance leg
        low_level_TG.policy.update_swing_stance()
        
        # do IK 
        for step in range(1, args.num_timestep_per_footstep+1):
            t_start = datetime.datetime.now()
            # check if footstep update/set a key stuff
            exp_variables = ru.get_variables(r)
            if exp_variables['update_z_action'][0]:
                z_action = np.array(exp_variables['z_action'])
                low_level_TG.policy.update_latent_action_params(state,z_action)
                exp_variables['update_z_action'] = [0]
                ru.set_variables(r, exp_variables)

            action = low_level_TG.get_action(state, step)
           
            state = env.step(action)
            

            velocity_stack[0].append(state['base_velocity'][0])
            velocity_stack[1].append(state['base_velocity'][1])
            velocity_record[0].append(state['base_velocity'][0])
            velocity_record[1].append(state['base_velocity'][1])
            postion_record[0].append((state['base_pos_x'][0]))
            postion_record[1].append((state['base_pos_y'][0]))
            postion_record[2].append((state['base_ori_euler'][2]))
            if len(velocity_stack[0])>STACK_LENGTH:
                velocity_stack[0].pop(0)
                velocity_stack[1].pop(0)
            
            t_end = datetime.datetime.now()
            t_diff = (t_end - t_start).total_seconds()
            time.sleep(max(0, 1.0/args.control_frequency - t_diff))


        # finish one step and update to high level 
        exp_variables['finish_one_step'] = [1]
        state['base_velocity'][0] = mean(velocity_stack[0])
        state['base_velocity'][1] = mean(velocity_stack[1])
        ru.set_state(r,state, exp_variables)
        exp_variables = ru.get_variables(r)


        if not exp_variables['not_finish_one_iter'][0]:
            exp_variables['not_finish_one_iter'][0] = [1]
            exp_variables['record_vel'] = velocity_record
            exp_variables['record_pos'] = postion_record
            ru.set_state(r,state, exp_variables)
            break

    
def main(args):
    r = redis.Redis(host = 'localhost', port=6379, db = 0)

    # initialize env
    env = daisy_API(sim=args.sim, realsense = True,render=args.render, logger = False)
    env.set_control_mode(args.control_mode)

    while True:
        input('New Iteration! Dont forget to check if the motors strategy are set correctly. Start client if its off')    
        exp_variables = ru.get_variables(r)
        if exp_variables['finish_exp'][0]:
            break
        run_LLTG_IK(env, args, r)

if __name__ == "__main__":
    args = parse_args()
    main(args)

         
    