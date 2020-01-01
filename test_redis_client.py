# Run on laptop/ robotdev


def collection_data():
     for i in range():
        # 
        # take current state and plan for next z_action and sent to daisy
        if args.test:  
        high_level_planning.plan_latent_action(state, target_speed)
        else:
        latent_action = high_level_planning.sample_latent_action(target_speed)
        latent_action_dict = {'HL_action': latent_action}
        r.set('z_action', latent_action_dict)

        # wait for one cycle
        while True:
            finish_cycle = r.get()
            if finish_cycle:
                break

def train_model():


def main():
    # initial initial redis
    r = redis.Redis(host='10.10.1.2', port=6379, db=0)
    state = 
    # define high level stuff
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_obs_dim, model_output_dim = np.size(utils.HL_obs(state)), np.size(utils.HL_delta_obs(state, state))
    utils.make_dir(args.save_dir)
    HL_replay_buffer = utils.ReplayBuffer(model_obs_dim, args.z_dim, model_output_dim, device,                 
                args.num_iters * args.num_latent_action_per_iteration)

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
        )


    # keys ={
    # finish one step key: for update z_action
    # experiment start/end key: if key =  1 then start experiment
    # 
    #
    # }
   


    # experiment ends and set the startexperiment key 0

