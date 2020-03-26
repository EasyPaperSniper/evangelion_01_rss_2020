import argparse

IK_setting = {
    'low_level_policy_type': 'IK',
    'IK_hidden_num' : 16,
    'model_update_steps' : 400,
    'z_dim': 3,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',default=1,type=int)
    parser.add_argument('--sim',default=1,type=int)
    parser.add_argument('--control_frequency',default=60,type=int)
    # parser.add_argument('--control_frequency',default=100,type=int) # for hardware
    parser.add_argument('--render',default=0,type=int)
    
    parser.add_argument('--control_mode',default='position',type=str)
    parser.add_argument('--num_iters',default= 200 ,type=int)
    parser.add_argument('--num_latent_action_per_iteration',default=20,type=int)
    parser.add_argument('--num_timestep_per_footstep',default=100,type=int)
    # parser.add_argument('--num_timestep_per_footstep',default=50,type=int) # for IK
    parser.add_argument('--model_hidden_num',default=64,type=int)
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--z_dim',default=1,type=int)
    parser.add_argument('--a_dim',default=18,type=int)
    parser.add_argument('--model_update_steps',default=2500,type=int)
    parser.add_argument('--model_lr',default=1e-3,type=float)
    parser.add_argument('--high_level_policy_type',default='random',type=str)
    parser.add_argument('--update_sample_policy',default=0,type=int)
    parser.add_argument('--update_sample_policy_lr',default=1e-3,type=float)
    parser.add_argument('--low_level_policy_type',default='NN',type=str)

    parser.add_argument('--update_low_level_policy',default=0,type=int)
    parser.add_argument('--update_low_level_policy_lr',default=1e-3,type=float)
    parser.add_argument('--start_training_sample_num',default=50,type=int)
    parser.add_argument('--low_level_buffer_size',default=10000,type=int)
    parser.add_argument('--save_dir',default='./save_data',type=str)


    args = parser.parse_args()


    if args.low_level_policy_type =='IK':
        args.update_low_level_policy = 0
        args.z_dim = 3
        args.a_dim = 18

    if args.high_level_policy_type =='random':
        args.update_sample_policy = 0
    
    if args.high_level_policy_type =='raibert':
        args.update_sample_policy = 0
        args.z_dim = 3
        args.test = 1

    if not args.test:
        args.render = 0
    else:
        args.num_iters = 5

    return args


if __name__ == '__main__':
    args = parse_args()