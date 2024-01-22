import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
import gymnasium as gym
import tempfile
from imitation.algorithms.dagger import SimpleDAggerTrainer


def train_expert(env):
    # note: use `download_expert` instead to download a pretrained, competent expert
    print("Training a expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
    )
    expert.learn(1_000)  # Note: change this to 100_000 to train a decent expert.
    return expert

def download_expert(env):
    print("Downloading a pretrained expert.")
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals-CartPole-v0",
        venv=env,
    )
    # expert = load_policy("ppo", env, path="ppo-CartPole-v1.zip")
    return expert


print("Loading expert demonstrations...")
rng = np.random.default_rng(0)

# venv = make_vec_env("CartPole-v1", post_wrappers=[lambda env, _: RolloutInfoWrapper(env)], rng=rng)
# expert = PPO.load("ppo_cartpole.zip", print_system_info=False)

venv = make_vec_env("LunarLander-v2", post_wrappers=[lambda env, _: RolloutInfoWrapper(env)], rng=rng)
expert = PPO.load("ppo_lunar.zip", print_system_info=False)

print("Rollouts...")
rollouts = rollout.rollout(
    expert,
    venv,
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    rng=rng,
)

bc_trainer = bc.BC(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    demonstrations=rollouts,
    rng=rng,
    device="cpu",
)

# reward, _ = evaluate_policy(
#     bc_trainer.policy,
#     venv,
#     n_eval_episodes=3,
#     render=False,
# )
# print(f"Reward before training: {reward}")
#
# print("Training a policy using Behavior Cloning...")
# bc_trainer.train(n_epochs=1)


# episode_rewards, episode_lengths = evaluate_policy(
#     bc_trainer.policy,
#     venv,
#     n_eval_episodes=3,
#     render=False,
#     return_episode_rewards=True,
# )
# print(f"Reward after training: {episode_rewards}")

# TODO: evaluation by gym.Env
# env = gym.make("LunarLander-v2", render_mode="human")
# obs, _ = env.reset()
# episode = 3
# for i in range(episode):
#     reward = 0
#     done = False
#     step = 0
#     while not done:
#         action, _states = bc_trainer.policy.predict(obs, deterministic=True)
#         obs, r, done, info, _ = env.step(action)
#         env.render()
#         reward += r
#         step += 1
#         if step % 100 == 0:
#             print(step)
#     print(f"episode: {i}, reward: {reward}")

# TODO: Dagger
with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    print(tmpdir)
    dagger_trainer = SimpleDAggerTrainer(
        venv=venv,
        scratch_dir=tmpdir,
        expert_policy=expert,
        bc_trainer=bc_trainer,
        rng=rng,
    )
    dagger_trainer.train(10000)

# env = gym.make("LunarLander-v2", render_mode="human")
reward, _ = evaluate_policy(
    dagger_trainer.policy,
    venv,
    n_eval_episodes=5,
    render=False,
    return_episode_rewards=True,
)
print(f"Reward of Dagger: {reward}")