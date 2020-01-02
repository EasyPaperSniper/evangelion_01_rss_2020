# run on Daisy
# 

import redis 


def send_state(r,state):
    r.set('state', state)
    key_dict = r.get('exp_keys')
    key_dict['finish_one_step'] = 1
    r.set('exp_keys', key_dict)


def run_LLTG(args, r, low_level_TG):
    # initialize robot
    if args.sim:
        state = motion_library.exp_standing(env)
        low_level_TG.reset(state)

    send_state(r, state)
    
    while True:
        footstep_dict = r.get('z_action')
        ft = footstep_dict['z_action']

        # update swing/stance leg
        low_level_TG.update_latent_action(state,latent_action)
        # do IK 
        for step in range(1, args.num_timestep_per_footstep+1):
            # check if footstep update/set a key stuff
            # update_latent_action
            action = low_level_TG.get_action(state, step)
            state = env.step(action)

        # finish one step and update to high level 

    
        # if end 


def main(args):

    r = redis.Redis(host = 'localhost', port=6379, db = 0)

    # initilize env

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

    while True:
        key_dict = r.get('exp_keys')
        if not key_dict['do_exp']:
            break
        run_LLTG(args, r, low_level_TG)
    
         
    