from typing import Type, List
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from torch import optim

import gym
from stable_baselines3.ppo import PPO
import argparse


# env = gym.make('LunarLander-v2')
# if PPO.load('ppo_lunar'):
#     print('Success')
random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# def create_mlp(input_dim: int, output_dim: int, architecture: List[int], squash=False,
#                activation: Type[nn.Module] = nn.ReLU) -> List[nn.Module]:
#     '''Creates a list of modules that define an MLP.'''
#     if len(architecture) > 0:
#         layers = [nn.Linear(input_dim, architecture[0]), activation()]
#     else:
#         layers = []
#
#     for i in range(len(architecture) - 1):
#         layers.append(nn.Linear(architecture[i], architecture[i + 1]))
#         layers.append(activation())
#
#     if output_dim > 0:
#         last_dim = architecture[-1] if len(architecture) > 0 else input_dim
#         layers.append(nn.Linear(last_dim, output_dim))
#
#     if squash:
#         # squashes output down to (-1, 1)
#         layers.append(nn.Tanh())
#
#     return layers
#
#
# def create_net(input_dim: int, output_dim: int, squash=False):
#     layers = create_mlp(input_dim, output_dim, architecture=[64, 64], squash=squash)
#     net = nn.Sequential(*layers)
#     print(net)
#     return net

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class BC:
    def __init__(self):
        self.net = MLP(state_dim=8, action_dim=4)
        self.loss_fn = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)

    def learn(self, env, states, actions, val_states, val_actions, n_steps=10000, val_steps=100, truncate=False):
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()
        val_states = torch.from_numpy(val_states).float()
        val_actions = torch.from_numpy(val_actions).float()

        loss_list = []
        reward_list = []
        val_loss_list = []
        for i in range(1, n_steps + 1):
            self.opt.zero_grad()
            outputs = self.net(states)
            loss = self.loss_fn(outputs, actions)
            loss.backward()
            self.opt.step()
            loss_list.append(loss.item())

            if i % val_steps == 0:
                with torch.no_grad():
                    val_outputs = self.net(val_states)
                    val_loss = self.loss_fn(val_outputs, val_actions)
                val_loss_list.append(val_loss.item())
                print("========== step: {}, val_loss: {} ==========".format(i, val_loss.item()))

            if i % 100 == 0:
                # reward_list.append(np.mean(eval_policy(argmax_policy(self.net), env, truncate)))
                reward_list.append(np.mean(eval_policy(self.net, env, truncate)))
        # return argmax_policy(self.net), loss_list, reward_list, val_loss_list
        return self.net, loss_list, reward_list, val_loss_list


# def argmax_policy(net):
#     def argmax_fn(state):
#         state = torch.from_numpy(state).float()
#         values = net(state)
#         max_value_index = torch.argmax(values)
#         return max_value_index
#
#     return argmax_fn


# def expert_policy(expert, s):
#     action = expert.predict(s)[0]
#     one_hot_action = np.eye(4)[action]
#     return one_hot_action

# def expert_rollout(expert, env, truncate=False):
#     expert_net = lambda s: expert_policy(expert, s)
#     return rollout(expert_net, env, truncate=truncate)


def expert_policy(net, env, truncate=False):
    states = []
    actions = []
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        states.append(ob.reshape(-1))
        ob_tensor = torch.from_numpy(np.array(ob)).float()
        if truncate:
            action = net(ob_tensor[:-3].float())
        else:
            action = net.predict(ob_tensor)[0]
        one_shot_action = np.eye(4)[action]
        if isinstance(action, torch.FloatTensor) or isinstance(action, torch.Tensor):
            action = action.detach().numpy()
        actions.append(one_shot_action.reshape(-1))
        ob, r, done, _ = env.step(action)
        total_reward += r

    states = np.array(states, dtype='float')
    actions = np.array(actions, dtype='float')
    return states, actions


def eval_policy(policy, env, truncate=False):
    done = False
    ob = env.reset()
    total_reward = 0
    while not done:
        ob_tensor = torch.from_numpy(np.array(ob)).float()
        if truncate:
            action = policy(ob[:-2])
        else:
            action = policy(ob_tensor)
            action = torch.argmax(action)

        # detach action and convert to np array
        if isinstance(action, torch.FloatTensor) or isinstance(action, torch.Tensor):
            action = action.detach().numpy()

        # step env and observe reward
        ob, r, done, _ = env.step(np.array([action]))
        total_reward += r

    return total_reward




def make_env():
    return gym.make("LunarLander-v2")


def get_expert():
    return PPO.load("ppo_lunar")


def get_expert_performance(env, expert):
    Js = []
    for _ in range(100):
        obs = env.reset()
        J = 0
        done = False
        hs = []
        while not done:
            action, _ = expert.predict(obs)
            obs, reward, done, info = env.step(action)
            hs.append(obs[0, 1])
            J += reward
        Js.append(J)
    ll_expert_performance = np.mean(Js)
    return ll_expert_performance


def train(truncate=False, n_steps=10000):
    # env = make_env()
    # expert = get_expert()
    expert = PPO.load("ppo_lunar", env=gym.make("LunarLander-v2"))
    env = expert.get_env()
    env.seed(random_seed)

    # performance = get_expert_performance(env, expert)
    # print('=' * 50)
    # print(f'Expert performance: {performance}')
    # print('=' * 50)

    # net + loss fn
    # if truncate:
    #     net = create_net(input_dim=5, output_dim=4)
    # else:
    #     net = create_net(input_dim=8, output_dim=4)
    # net = MLP(state_dim=8, action_dim=4)
    # loss_fn = nn.CrossEntropyLoss()


    # TODO: train BC
    # Things that need to be done:
    # - Roll out the expert for X number of trajectories (a standard amount is 10).
    # - Create our BC learner, and train BC on the collected trajectories.
    # - It's up to you how you want to structure your data!
    # - Evaluate the argmax_policy by printing the total rewards.

    # Create our data so that X number of expert trajectories are rolled out to accumulate data
    n_trajectories = 10
    expert_states, expert_actions = [], []
    for i in range(n_trajectories):
        states_traj, actions_traj = expert_policy(expert, env)
        expert_states.append(states_traj)
        expert_actions.append(actions_traj)
    # Concatenate the trajectory data into single arrays
    expert_states = np.concatenate(expert_states, axis=0)
    expert_actions = np.concatenate(expert_actions, axis=0)

    # create our validation data
    val_states, val_actions = [], []
    for i in range(n_trajectories):
        states_traj, actions_traj = expert_policy(expert, env)
        val_states.append(states_traj)
        val_actions.append(actions_traj)

    # Concatenate the trajectory data into single arrays
    val_states = np.concatenate(val_states, axis=0)
    val_actions = np.concatenate(val_actions, axis=0)

    # Create our BC learner, and train BC on the collected trajectories.
    bc_learner = BC()
    trained_policy, loss_list, reward_list, val_loss_list = bc_learner.learn(env, expert_states, expert_actions,
                                                                             val_states, val_actions, n_steps=n_steps,
                                                                             val_steps=100, truncate=truncate)

    # Evaluate the argmax_policy by printing the total rewards.
    reward = 0
    for i in range(10):
        r = eval_policy(trained_policy, env, truncate=truncate)
        print(r)
        reward += r
    print("BC Performance: %.2f" % (reward / 10))
    # Plot the reward over time
    plt.figure(1)
    plt.plot(reward_list)
    plt.title("Reward over time")
    plt.xlabel("Training steps")
    plt.ylabel("Reward")
    # plt.show()

    # Plot the loss over time
    plt.figure(2)
    plt.plot(loss_list)
    plt.title("Validation Loss over time")
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.show()

    # Plot the validation loss
    # plt.plot(val_loss_list)
    # plt.title("Validation Loss")
    # plt.xlabel("Training steps")
    # plt.ylabel("Loss")
    # plt.show()



truncate = False
n_steps = 1000

train(truncate, n_steps)

# ========== step: 100, val_loss: 1.1882914304733276 ==========
# ========== step: 200, val_loss: 1.1260299682617188 ==========
# ========== step: 300, val_loss: 1.1393760442733765 ==========
# ========== step: 400, val_loss: 1.1166514158248901 ==========
# ========== step: 500, val_loss: 1.0420212745666504 ==========
# ========== step: 600, val_loss: 0.9230090379714966 ==========
# ========== step: 700, val_loss: 0.8521901965141296 ==========
# ========== step: 800, val_loss: 0.8257066607475281 ==========
# ========== step: 900, val_loss: 0.8153381943702698 ==========
# ========== step: 1000, val_loss: 0.8109278082847595 ==========
# [149.35661]
# [-0.6477952]
# [210.1513]
# [185.44724]
# [186.53896]
# [179.29358]
# [236.06322]
# [234.31137]
# [142.14305]
# [220.86368]
# BC Performance: 174.35