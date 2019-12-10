import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name',default='AntBulletEnv-v0')

    retun parser
    
if __name__ == '__main__':
    parse_args()
   