import gymnasium as gym
from stable_baselines3 import PPO
import torch as th
import os


models_dir = "models/PPO_policy_kwargs"
logdir = 'logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# Custom actor (pi) and value function (vf) networks
# of two layers of size 32 each with Relu activation function
# Note: an extra linear layer will be added on top of the pi and the vf nets, respectively
# src: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
policy_kwards = dict(activation_fn=th.nn.ReLU,
					 net_arch=dict(pi=[32,32], vf=[32,32]))

env = gym.make("FrozenLake-v1", map_name='4x4', is_slippery=True, render_mode="rgb_array")
env.reset()


model = PPO('MlpPolicy', env, verbose=1, policy_kwards=policy_kwards, tensorboard_log=logdir )

TIMESTEPS = 10000
for i in range (1,30):
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_policy_kwargs")
	model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()
