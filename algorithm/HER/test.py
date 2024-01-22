import torch
import os
import gymnasium as gym
import numpy as np

from actor_critic import *


def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs


def test_agent(args):
    #path = os.path.join(args.model_dir, os.path.join(args.env_name,"actor_target_her.pth"))
    model = "/home/kavin/Documents/PycharmProjects/RL/algorithm/HER/saved_models/FetchPushDense-v2/model_scratch_again_puck.pt"
    o_mean, o_std, g_mean, g_std, model = torch.load(model, map_location=lambda storage, loc: storage)

    #print("[*] Model loaded from: ",path)

    #o_mean, o_std, g_mean, g_std, model = torch.load(model, map_location=lambda storage, loc: storage)

    env = gym.make(args.env_name, render_mode='human')
    observation, _ = env.reset()
    #print("Initial observation: ", observation)

    env_params = {
        'obs_dim' : observation['observation'].shape[0], #(25,)
        'goal_dim': observation['desired_goal'].shape[0],  #(3,)
        'action_dim': env.action_space.shape[0], #(4,)
        'max_action' : env.action_space.high[0], # high : [1,1,1,1] low: [-1,-1,-1,-1]
    }

    # create instance of actor for testing model
    actor = Actor(env_params, True)
    #actor.load_state_dict(torch.load(path))
    actor.load_state_dict(model)
    actor.eval()

    for episode in range(args.test_episodes):
        observation, _ = env.reset()
        obs = observation['observation']
        #obs = torch.tensor(obs, dtype=torch.float32)
        goal = observation['desired_goal']
        #print(obs)
        #print(goal)
        # inputs = np.concatenate([obs, goal])
        # inputs = torch.tensor(inputs, dtype=torch.float32)#.unsqueeze(0)
        for step in range(env._max_episode_steps):
            # env.render()
            # inputs = np.concatenate([obs, goal])
            # inputs = torch.tensor(inputs, dtype=torch.float32)#.unsqueeze(0)
            #print(inputs)
            state = process_inputs(obs, goal, o_mean, o_std, g_mean, g_std, args)
            # get actions for current state
            with torch.no_grad():
                actions = actor(state).cpu().numpy().squeeze()
            # carry out action
            obs_new, reward, terminated, truncated, info = env.step(actions)
            # get next state
            #print(obs_new==obs)
            obs = obs_new['observation']
            #obs = torch.tensor(obs, dtype=torch.float32)
        print("Episode number : {} Reward : {} Success : {}".format(episode, reward, info['is_success']))

    env.close()
