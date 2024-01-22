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
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm

SEED = 42
print("Loading expert demonstrations...")
rng = np.random.default_rng(0)
venv = make_vec_env("LunarLander-v2", post_wrappers=[lambda env, _: RolloutInfoWrapper(env)], rng=rng)
expert = PPO.load("ppo_lunar.zip", print_system_info=False)

print("Rollouts...")
rollouts = rollout.rollout(
    expert,
    venv,
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    rng=rng,
)

learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=5,
    seed=SEED,
)

reward_net = BasicRewardNet(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    normalize_input_layer=RunningNorm,
)

gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True,    # Some algorithms are biased towards shorter or longer episodes, which may significantly confound results.
)

venv.seed(SEED)
learner_reward_before_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)

gail_trainer.train(200_000)
venv.seed(SEED)
learner_reward_after_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)

print(f"Reward after training: {np.mean(learner_reward_before_training)}")
print(f"Reward after training: {np.mean(learner_reward_after_training)}")