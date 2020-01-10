
import json
import numpy as np

state_dict = ['base_pos_x', 'base_pos_y', 'base_pos_z', 'base_ori_euler', 'base_velocity', 'j_pos', 'j_vel']

def set_variables(r, exp_variables):
    exp_variables = json.dumps(exp_variables)
    r.set('exp_variables', exp_variables)
    return exp_variables

def get_variables(r):
    exp_variables = r.get('exp_variables')
    exp_variables = json.loads(exp_variables)
    return exp_variables

def wrap_state(state,exp_variables):
    for index in state_dict:
        exp_variables[index] = state[index].tolist()
    return exp_variables

def unwarp_state(exp_variables):
    state ={}
    for index in state_dict:
        state[index] = np.array(exp_variables[index])
    return state

def get_state(r):
    exp_variables = get_variables(r)
    state = unwarp_state(exp_variables)
    return state

def set_state(r, state, exp_variables):
    exp_variables = wrap_state(state, exp_variables)
    set_variables(r, exp_variables)
    return exp_variables

def wait_for_key(r,key):
    print(' Wait for' + key)
    while True:
        exp_variables = get_variables(r)
        if exp_variables[key][0]:
            exp_variables[key] = [0]
            set_variables(r, exp_variables)
            print('Finished ' + key)
            return exp_variables




    