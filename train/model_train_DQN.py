import gymnasium as gym
from stable_baselines3 import DQN
import torch as th
import os


models_dir = "../models/DQN_policy_kwargs"
logdir = '../logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# An extra linear layer will be added on top of the layers specified in net_arch with two
# hidden layers with 32 neurons each.
# src: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=[32, 32]
)

env = gym.make("FrozenLake-v1", map_name='4x4', is_slippery=True, render_mode="rgb_array")
env.reset()


model = DQN('MlpPolicy', env, verbose=1,policy_kwargs=policy_kwargs, tensorboard_log=logdir )

TIMESTEPS = 10000
for i in range (1,30):
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN_policy_kwargs")
	model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()
