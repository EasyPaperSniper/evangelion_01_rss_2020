

import daisy_API
import daisy_hardware.motion_library as motion_library
import high_level_planning_model as HLPM
import low_level_traj_gen as LLTG
from args_all import parse_args 


def main(args):
    # define env & high level planning part & low level trajectory generator & replay buffer for HLP
    # initialize logger
    env = daisy_API(sim=args.sim, render=args.render)
    env.set_control_mode(args.control_mode)
    state = env.reset()

    high_level_planning = HLPM.high_level_planning()
    low_level_TG = LLTG.low_level_TG( )

    for _ in range(args.num_iters):
        # reset robot to stand 
        if args.sim:
            state = motion_library.exp_standing(env)

        for index_latent_action in range(args.num_latent_action_per_iteration):
            # generate foot footstep position. If test, the footstep comes from optimization process
            pre_com_state = state
            if args.test:
                latent_action = high_level_planning.plan_latent_action()
            else:
                latent_action = high_level_planning.sample_latent_action()
            # update LLTG (target footstep position and stance & swing leg)
            low_level_TG.update(latent_action)
            for step in range(1, args.num_timestep_per_footstep+1):
                action = XXXX.get_action(state, step)
                state = env.step(action)

            post_com_state = state
            # Judge if robot still alive
            if robot_die(post_com_state):
                break
            # replay buffer collect data collect data 
            # [pre_com_state, post_com_state, latent_action]

        for _ in range(args.model_update_per_iter):
            # model update
            
    


if __name__ == "__main__":
    main(parse_args)



