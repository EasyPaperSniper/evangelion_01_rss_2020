import time
import math

import numpy as np
import matplotlib.pyplot as plt
import daisy_hardware.motion_library as motion_library
from daisy_API import daisy_API
import daisy_kinematics

from high_level_planning_model import raibert_footstep_policy
from low_level_traj_gen import IK_traj_generator


def main(test_arg):
    # initialize data structure record time to measure the frequency
    tgt_pos = np.zeros((6,3,test_arg['total_step']*test_arg['stance_time']))
    actual_pos = np.zeros((6,3,test_arg['total_step']*test_arg['stance_time']))

    # initialize the configuration
    env = daisy_API(sim=test_arg['sim'], render=True, logger = False)
    env.set_control_mode('position')
    
    state = env.reset()
    if test_arg['sim']:
        state = motion_library.exp_standing(env)
    else:
        state = motion_library.demo_standing(env, shoulder = 0.5, elbow = 1.1)

    # initialize controllers
    high_level_planning = raibert_footstep_policy(stance_duration=test_arg['stance_time'], target_speed=np.array(test_arg['target_speed']), control_frequency=test_arg['control_frequency'])
    low_level_planning =  IK_traj_generator(state)
    

    # run controller for several steps record data
    for index_step in range(test_arg['total_step']):
        latent_action = high_level_planning.plan_latent_action(state)
        low_level_planning.update_latent_action(state,latent_action)
        
        for i in range(1, test_arg['stance_time']+1):
            phase = float(i)/test_arg['stance_time']
            action = low_level_planning.get_action(state, phase)
            # TODO: implement simplified non blocking stuff
            state = env.step(action)
            time.sleep(0.01)
            # calculate current XYZ
            foot_com_cur = daisy_kinematics.FK_CoM2Foot(state['j_pos'])
            foot_com_tgt = low_level_planning.des_foot_position_com
            
            for j in range(6):
                for k in range(3):
                    tgt_pos[j][k][index_step*test_arg['stance_time']+i-1] = foot_com_tgt[j][k] 
                    actual_pos[j][k][index_step*test_arg['stance_time']+i-1] = foot_com_cur[j][k] 

    # save data + load data + plotting
    np.save('./save_data/test_tra/tgt_tra_'+str(test_arg['stance_time']) + '.npy', tgt_pos)
    np.save('./save_data/test_tra/actual_tra_'+str(test_arg['stance_time']) + '.npy', actual_pos)

    tgt_pos = np.load('./save_data/test_tra/tgt_tra_'+str(test_arg['stance_time']) + '.npy')
    actual_pos = np.load('./save_data/test_tra/actual_tra_'+str(test_arg['stance_time']) + '.npy')

    leg_index = 0
    axis = range(np.shape(tgt_pos)[2])
    plt.rcParams['figure.figsize'] = (8, 10)
    fig, (ax1,ax2,ax3) = plt.subplots(3,1)
    ax1.plot(axis, tgt_pos[leg_index][0], color='dodgerblue')
    ax1.plot(axis, actual_pos[leg_index][0], color='deeppink')
    ax1.set_title('X tracking')
    ax2.plot(axis, tgt_pos[leg_index][1], color='dodgerblue')
    ax2.plot(axis, actual_pos[leg_index][1], color='deeppink')
    ax2.set_title('Y tracking')
    ax3.plot(axis, tgt_pos[leg_index][2], color='dodgerblue')
    ax3.plot(axis, actual_pos[leg_index][2], color = 'deeppink')
    ax3.set_title('Z tracking')
    plt.show()

    



if __name__ == "__main__":
    test_arg = {
        'sim':False,
        'target_speed': [0.0, 0.8, 0.0],
        'stance_time': 35,
        'control_frequency': 100,
        'total_step':15
        
    }
    main(test_arg)
