import gymnasium as gym
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# env = gym.make("LunarLander-v2", render_mode="rgb_array")
env = gym.make("CartPole-v1", render_mode="rgb_array")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=int(1e5), progress_bar=True)
model.save("ppo_cartpole")
del model

# model = PPO.load("ppo_lunar", env=env)
model = PPO.load("ppo_cartpole", env=env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"mean_reward: {mean_reward}, std_reward: {std_reward}")

vec_env = model.get_env()
obs = vec_env.reset()
episode = 5
for i in range(episode):
    reward = 0
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, r, done, info = vec_env.step(action)
        vec_env.render("human")
        reward += r[0]
    print(f"episode: {i}, reward: {reward}")