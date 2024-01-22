import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
import easydict
from multiprocessing_env import SubprocVecEnv


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value


# class A2C:
#     def __init__(self, n_states, n_actions, cfg):
#         self.gamma = cfg.gamma
#         self.device = cfg.device
#         self.model = ActorCritic(n_states, n_actions, cfg. hidden_size).to(self.device)
#         self.optimizer = optim.Adam(self.model.parameters())
#
#     def compute_returns(self, next_value, rewards, masks):
#         R = next_value
#         returns = []
#         for step in reversed(range(len(rewards))):
#             R = rewards[step] + self.gamma * R * masks[step]
#             returns.insert(0, R)
#         return returns
# def make_envs(env_name):
#     def _thunk():
#         env = gym.make(env_name)
#         return env
#     return _thunk()

def test_env(env, model, cfg, vis=False):
    state, _ = env.reset()
    if vis:
        env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(cfg.device)
        dist, _ = model(state)
        next_state, reward, done, _, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis:
            env.render()
        total_reward += reward
    return total_reward


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def train(cfg):
    env = gym.make(cfg.env_name)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    model = ActorCritic(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
    optimizer = optim.Adam(model.parameters())
    step_idx = 0
    test_rewards = []
    test_ma_rewards = []
    state, _ = env.reset()
    while step_idx < cfg.max_steps:
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        for _ in range(cfg.n_steps):
            state = torch.from_numpy(state).float().unsqueeze(0).to(cfg.device)
            dist, value = model(state)
            action = dist.sample()
            next_state, reward, done, _, _ = env.step(action.cpu().numpy()[0])
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(cfg.device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(cfg.device))
            state = next_state
            step_idx += 1
            if step_idx % 2 == 0:
                test_reward = np.mean([test_env(env, model, cfg) for _ in range(10)])
                print(f'step_idx: {step_idx}, test_reward: {test_rewards}')
                test_rewards.append(test_reward)
                if test_ma_rewards:
                    test_ma_rewards.append(0.9 * test_ma_rewards[-1] + 0.1 * test_reward)
                else:
                    test_ma_rewards.append(test_reward)
            if done:
                break

        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(cfg.device)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks)
        print(next_value)
        print(rewards)
        print(masks)
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Finish training')
    return test_rewards, test_ma_rewards


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# prob = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device)
# dist = Categorical(prob)
# print(dist.sample().cpu().numpy()[0])

cfg = easydict.EasyDict({
    'env_name': 'CartPole-v1',
    'max_steps': 200,
    'n_steps': 5,
    'gamma': 0.99,
    'lr': 1e-3,
    'hidden_dim': 64,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 'device': 'cpu'
})

rewards, ma_rewards = train(cfg)

# env = gym.make(cfg.env_name)
# n_states = env.observation_space.shape[0]
# print(n_states)
# n_actions = env.action_space.n
# print(n_actions)
# model = ActorCritic(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
# state, _ = env.reset()


# state = torch.from_numpy(state).float().unsqueeze(0).to(cfg.device)
# action = torch.tensor([0], device='cuda:0')
# print(action.cpu().numpy())
# print(model(state))


# log_probs = []
# values = []
# rewards = []
# masks = []
# entropy = 0
# state = torch.from_numpy(state).float().unsqueeze(0).to(cfg.device)
# dist, value = model(state)
# action = dist.sample()
# next_state, reward, done, _, _ = env.step(action.cpu().numpy()[0])
# log_prob = dist.log_prob(action)
# entropy += dist.entropy().mean()
# log_probs.append(log_prob)
# values.append(value)
# print(reward)
# r = torch.FloatTensor([reward]).unsqueeze(1)
# print(r)
# print(r.shape)
#
# print(done)
# d = torch.FloatTensor(1 - done).unsqueeze(1)
# print(d)
# print(d.shape)

# rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(cfg.device))
# masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(cfg.device))
# state = next_state