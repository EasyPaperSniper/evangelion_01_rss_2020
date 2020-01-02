# run on Daisy
# 

import redis 


def send_state(r,state):
    r.set('state', state)
    key_dict = r.get('exp_keys')
    key_dict['finish_one_step'] = 1
    r.set('exp_keys', key_dict)
    return r


def run_LLTG_IK(args, r, low_level_TG):
    # initialize robot
    if args.sim:
        state = motion_library.exp_standing(env)
        low_level_TG.reset(state)

    r = send_state(r, state)
    
    while True:
        # update swing/stance leg
        low_level_TG.policy.update_swing_stance()
        
        # do IK 
        for step in range(1, args.num_timestep_per_footstep+1):
            # check if footstep update/set a key stuff
            key_dict = r.get('exp_keys')
            if key_dict['updated_z_action']:
                footstep_dict = r.get('z_action')
                z_action = footstep_dict['z_action']
                low_level_TG.policy.update_latent_action_params(state,z_action)
                key_dict['update_z_action'] = 0
                r.set('exp_keys', key_dict)

            action = low_level_TG.get_action(state, step)
            state = env.step(action)

        # finish one step and update to high level 
        r = send_state(r, state)
        key_dict = r.get('exp_keys')
        
        if not key_dict['do_one_iter']:
            break


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
    
         
    