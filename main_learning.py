
import numpy as np

import daisy_API
import daisy_hardware.motion_library as motion_library
import high_level_planning_model as HLPM
import low_level_traj_gen as LLTG
from args_all import parse_args 
import utils


def main(args):
    # define env & high level planning part & low level trajectory generator & replay buffer for HLP
    # initialize logger
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = daisy_API(sim=args.sim, render=args.render)
    env.set_control_mode(args.control_mode)
    state = env.reset()

    HL_replay_buffer = utils.ReplayBuffer(args.HL_obs_dim, args.z_dim, device, args.HL_buffer_size)

    high_level_planning = HLPM.high_level_planning(
        HL_obs_dim,
        z_dim,
        HL_output_dim,
        model_hidden_num,
        limits,
        batch_size,
        sample_num,
        model_lr,
        policy_type = 'random',
    )
    
    low_level_TG = LLTG.low_level_TG( )

    for _ in range(args.num_iters):
        # reset robot to stand 
        if args.sim:
            state = motion_library.exp_standing(env)

        for index_latent_action in range(args.num_latent_action_per_iteration):
            # generate foot footstep position. If test, the footstep comes from optimization process
            pre_com_state = np.copy(state)
            if args.test:
                latent_action = high_level_planning.plan_latent_action()
            else:
                latent_action = high_level_planning.sample_latent_action()
            
            # update LLTG (target footstep position and stance & swing leg)
            low_level_TG.update(latent_action)
            
            for step in range(1, args.num_timestep_per_footstep+1):
                action = low_level_TG.get_action(state, step)
                state = env.step(action)

            post_com_state = np.copy(state)
            # Judge if robot still alive
            if utils.check_robot_dead(state):
                break
            else:
                # TODO: process pre_com_state and post_com_state
                HL_replay_buffer.add(pre_com_state, latent_action, 0, post_com_state, 1) # reward = 0; done = 1

        for _ in range(args.model_update_per_iter):
            # model update
            high_level_planning.update_model(HL_replay_buffer)

    


if __name__ == "__main__":
    main(parse_args)



