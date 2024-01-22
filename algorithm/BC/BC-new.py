import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch import optim
import gymnasium as gym
from stable_baselines3.ppo import PPO
import matplotlib.pyplot as plt

################################## set device ##################################
print("============================================================================================")
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

################################## set random_seed ##################################
random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)

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
    def __init__(self, input_dim, output_dim):
        self.net = MLP(input_dim, output_dim)
        print(self.net)
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)
        self.loss_fn = nn.CrossEntropyLoss()

    def learn(self, env, states, actions, val_states, val_actions, n_steps, val_steps):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        val_states = torch.FloatTensor(val_states)
        val_actions = torch.FloatTensor(val_actions)

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
                reward = eval_policy(env, self.net)
                reward_list.append(np.mean(reward))

        return loss_list, reward_list, val_loss_list


def get_expert_performance(env, expert):
    Js = []
    for _ in range(100):
        obs = env.reset()
        J = 0
        done = False
        while not done:
            action, _ = expert.predict(obs)
            obs, reward, done, info = env.step(action)
            J += reward
        Js.append(J)
    ll_expert_performance = np.mean(Js)
    return ll_expert_performance


def expert_policy(env, expert):
    states = []
    actions = []
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        states.append(ob.reshape(-1))
        ob_tensor = torch.FloatTensor(ob)
        action, _ = expert.predict(ob_tensor)
        one_hot_action = np.eye(4)[action]   # action: [2] --> action[0]: 2 --> [0, 0, 1, 0]
        if isinstance(action, torch.FloatTensor) or isinstance(action, torch.Tensor):
            action = action.detach().numpy()

        actions.append(one_hot_action.reshape(-1))  # [[0, 1, 0, 0]] --> [0, 1, 0, 0]
        ob, r, done, _ = env.step(action)
        total_reward += r

    states = np.array(states, dtype='float')    # list of array: [array([1, 2]), array([10, 20])] --> array: [[1, 2], [10, 20]]
    actions = np.array(actions, dtype='float')
    return states, actions


def eval_policy(env, policy):
    done = False
    ob = env.reset()
    total_reward = 0
    while not done:
        input = torch.FloatTensor(ob)
        output = policy(input)
        action = torch.argmax(output).detach().numpy()
        ob, r, done, _ = env.step(np.array([action]))   # 1 --> array([1])
        total_reward += r

    return total_reward


def train(n_steps, val_steps):
    expert = PPO.load("ppo_lunar", env=gym.make("LunarLander-v2"))
    env = expert.get_env()
    env.seed(random_seed)

    # performance = get_expert_performance(env, expert)
    # print('=' * 50)
    # print(f'Expert performance: {performance}')
    # print('=' * 50)

    n_trajectories = 10
    expert_states, expert_actions = [], []
    for i in range(n_trajectories):
        states_traj, actions_traj = expert_policy(env, expert)
        expert_states.append(states_traj)
        expert_actions.append(actions_traj)
    # Concatenate the trajectory data into single arrays
    expert_states = np.concatenate(expert_states, axis=0)
    expert_actions = np.concatenate(expert_actions, axis=0)

    # create our validation data
    val_states, val_actions = [], []
    for i in range(n_trajectories):
        states_traj, actions_traj = expert_policy(env, expert)
        val_states.append(states_traj)
        val_actions.append(actions_traj)

    # Concatenate the trajectory data into single arrays
    val_states = np.concatenate(val_states, axis=0)
    val_actions = np.concatenate(val_actions, axis=0)

    bc_learner = BC(input_dim=8, output_dim=4)
    loss_list, reward_list, val_loss_list = bc_learner.learn(env, expert_states, expert_actions,
                                                             val_states, val_actions,
                                                             n_steps=n_steps, val_steps=val_steps)

    reward = 0
    for i in range(10):
        reward += eval_policy(env, bc_learner.net)
    print("BC Performance: %.2f" % (reward / 10))

    plt.figure(1)
    plt.plot(reward_list)
    plt.title("Reward over time")
    plt.xlabel("Training steps")
    plt.ylabel("Reward")

    plt.figure(2)
    plt.plot(loss_list)
    plt.title("Validation Loss over time")
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.show()


train(n_steps=1000, val_steps=100)