
import json
import numpy as np


def set_variables(r, exp_variables):
    exp_variables = json.dumps(exp_variables)
    r.set('exp_variables', exp_variables)

def get_state(r):
    exp_variables = r.get('exp_variables')
    exp_variables = json.loads(exp_variables)
    state = np.zeros(0)
    return state

def wait_for_one_step(r):
    while True:
        exp_variables = r.get('exp_variables')
        exp_variables = json.loads(exp_variables)
        if exp_variables['finish_one_step'][0]:
            exp_variables['finish_one_step'] = [0]
            set_variables(r, exp_variables)
            break