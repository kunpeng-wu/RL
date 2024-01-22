
import gym
import numpy as np

env = gym.make('FetchPush-v1', render_mode="human")
# print(env.action_space)
# print(env.action_space.high)
# print(env.observation_space)
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
# print(state_dim)
# print(action_dim)
print(max_action)
print(env._max_episode_steps)
env.action_space.seed(42)

observation, info = env.reset(seed=42)

step = 0
# for _ in range(100):
while True:
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(terminated, truncated)
    step += 1
    if terminated or truncated:
        # observation, info = env.reset()
        break
print(step)
env.close()


# state = np.zeros((10, 3))
# ptr = 0
# size = 0
# for _ in range(3):
#     state[ptr] = np.random.randn(3)
#     ptr += 1
#     size += 1
# print(state)
#
# ind = np.random.randint(0, size, size=5)
# print(ind)
# print(state[ind])
