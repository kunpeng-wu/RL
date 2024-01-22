import gymnasium as gym

env = gym.make("FetchPushDense-v2", render_mode='human', max_episode_steps=100)
print(env._max_episode_steps)   # 100
print(env.action_space)
print(env.action_space.high)
obs_spec = env.observation_space
print(obs_spec['achieved_goal'])
print(obs_spec['desired_goal'])
print(obs_spec['observation'])

obs, _ = env.reset()
print(obs['achieved_goal'])
print(obs['desired_goal'])
print(obs['observation'])

for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    # print(obs['desired_goal'])
    # print(obs['achieved_goal'])
    # print(info['is_success'])
    if terminated or truncated:
        break
env.close()