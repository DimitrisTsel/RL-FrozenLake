import gymnasium as gym
from stable_baselines3 import PPO
import torch as th
import os
from customRewardWrapper import CustomRewardWrapper

models_dir = "../models/PPO_custom_reward"
logdir = '../logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)


policy_kwargs = dict(activation_fn=th.nn.ReLU,
					 net_arch=dict(pi=[32,32], vf=[32,32]))

env = gym.make("FrozenLake-v1", map_name='4x4', is_slippery=True)
env = CustomRewardWrapper(env)
env.reset()

model = PPO('MlpPolicy', env, verbose=1,
            learning_rate=0.001,
            ent_coef=0.01,
            gae_lambda=0.95,
            n_steps=2048,
            policy_kwargs=policy_kwargs,
            tensorboard_log=logdir)

TIMESTEPS = 10000
for i in range (1,30):
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_custom_reward")
	model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()
