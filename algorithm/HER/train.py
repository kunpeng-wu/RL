import torch
import gymnasium as gym
import os
# from vanilla_ddpg import *
# from ddpg_with_her import *
from ddpg_her_normalizn import *
import argparse

def train_agent(args):
    model_dir = args.model_dir + '/' + args.env_name
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    env = gym.make(args.env_name)
    observation, _ = env.reset()

    # print("Initial observation: ", observation)

    env_params = {

        # for hopper v2
        # 'obs_dim' : observation.shape[0], #11
        # for fetch slide
        'obs_dim': observation['observation'].shape[0],  # (25,)
        'goal_dim': observation['desired_goal'].shape[0],  # (3,)
        'action_dim': env.action_space.shape[0],  # (4,)
        'max_action': env.action_space.high[0],  # high : [1,1,1,1] low: [-1,-1,-1,-1]
    }
    print(env_params)

    if args.her:
        ddpg_agent = DDPG_HER_N(args, env, env_params)
    # else:
    #     ddpg_agent = DDPG(args, env, env_params)

    ddpg_agent.train()
    env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='FetchPushDense-v2', help='Fetch environment name')
    args = parser.parse_args()
    train_agent(args)