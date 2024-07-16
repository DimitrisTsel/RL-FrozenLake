# FrozenLake Reinforcement Learning Project

This project demonstrates the training and evaluation of RL agents to navigate through a slippery frozen lake to
reach the goal safely.
We use the FrozenLake-v1 environment and the DQN and PPO algorithms from Stable-Baselines3. Additionally, it provides an API to interact with the environment.

## Project Structure
```
.
├── logs
├── models
├── src
│ ├── client.py
│ └── FrozenLakeAPI_structure.py
└── train
| ├── customRewardWrapper.py
| ├── model_train_DQN.py
│ ├── model_train_PPO.py
│ └── model_train_PPO_custom_reward_wrapper.py
└── requirements.txt
```

- `logs/`: Directory to store TensorBoard logs.
- `models/`: Directory to save trained models.
- `src/`: Source code for the API and client.
  - `client.py`: Script to evaluate and test models using the provided API.
  - `FrozenLakeAPI_structure.py`: Flask application to create and interact with FrozenLake environments.
- `train/`: Scripts to train DQN and PPO models.
  - `customCrewardWrapper.py`: Contains the CustomRewardWrapper class to change the reward logic.
  - `model_train_DQN.py`: Script to train a DQN model.
  - `model_train_PPO.py`: Script to train a PPO model.
  - `model_train_PPO_custom_reward_wrapper.py`: Script to train the PPO model with the custom reward logic.

## Getting Started

### Environment Setup

- Python 3.8+
- create an isolated python virtual environment
  - ` python -m venv venv`
  - `source venv/bin/activate`
- Install dependencies using `pip`:


  `pip install -r requirements.txt`

### Train the models
#### DQN  and PPO models
To train a DQN model, run: `python train/model_train_DQN.py`,
to train a PPO model run: `python train/model_train_PPO.py`,
and to train a PPO model with custom reward logic, run: `model_train_PPO_custom_reward_wrapper.py`.

This will save the trained models either on models/DQN or on models/PPO directory, and log training details under the logs directory.
**Description:**
  - The scripts use the DQN and PPO algorithms from Stable-Baselines3 to train a model on the FrozenLake-v1 environment.
  - A custom policy architecture is also defined for both DQN and PPO models. An extra linear layer is added on top of the layers specified in net_arch with two hidden layers with 32 neurons each and retrained the models.  
    ref: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
  - A custom reward wrapper created for modifying the reward logic of the environment. By overriding the reward method, the CustomRewardWrapper modified the reward logic assigning a penalty of -1 for each step taken and a reward
    of 10 forn reaching the goal. Then, the agent retrained, but only with the PPO algorithm.
  - Multiple checkpoints are saved (10000*30) during the training of every algorithm-model to facilitate experimentation and evaluation throughout the training process.
  - Training details are logged to TensorBoard for visualization and analysis.
    To view the TensorBoard logs, run: `tensorboard --logdir=logs`

### Running the API
To start the FrozenLake API, run: `python src/FrozenLakeAPI_structure.py`

The server host address is  http://localhost:5005.

### API Endpoints
- [GET]  /new_game: Creates a new game and returns a unique game_id using uuid.
- [POST] /reset: Resets the environment for the specified game_id.
- [POST] /step: Takes a step in the environment for the specified game_id using the provided action.

### Evaluating and Testing Models
To evaluate and test the models, run: `python src/client.py`
This script will:
- Load the best-trained models.
- Evaluate each model using evaluate_policy.
- Test each model by running multiple episodes and calculating success_rate and average_reward.

### Algorithm comparison
An example of the output is the below:
```
2024-07-17 17:50:31,163 - INFO - Evaluating DQN model...
2024-07-17 17:50:32,465 - INFO - Mean reward: 0.73 Std reward:0.4439594576084623
2024-07-17 17:50:32,465 - INFO - Testing DQN model...
2024-07-17 17:50:50,371 - INFO - DQN - Success rate: 0.81
2024-07-17 17:50:50,371 - INFO - DQN - Average reward: 0.81
2024-07-17 17:50:50,371 - INFO - Evaluating DQN_custom_policy model...
2024-07-17 17:50:51,510 - INFO - Mean reward: 0.75 Std reward:0.4330127018922193
2024-07-17 17:50:51,511 - INFO - Testing DQN_custom_policy model...
2024-07-17 17:51:09,407 - INFO - DQN_custom_policy - Success rate: 0.73
2024-07-17 17:51:09,407 - INFO - DQN_custom_policy - Average reward: 0.73
2024-07-17 17:51:09,407 - INFO - Evaluating PPO model...
2024-07-17 17:51:11,795 - INFO - Mean reward:  0.8 Std reward:0.4
2024-07-17 17:51:11,795 - INFO - Testing PPO model...
2024-07-17 17:51:29,713 - INFO - PPO - Success rate: 0.83
2024-07-17 17:51:29,713 - INFO - PPO - Average reward: 0.83
2024-07-17 17:51:29,713 - INFO - Evaluating PPO_custom_policy model...
2024-07-17 17:51:31,753 - INFO - Mean reward:  0.78 Std reward:0.41424630354415964
2024-07-17 17:51:31,753 - INFO - Testing PPO_custom_policy model...
2024-07-17 17:51:50,769 - INFO - PPO_custom_policy - Success rate: 0.78
2024-07-17 17:51:50,769 - INFO - PPO_custom_policy - Average reward: 0.78
2024-07-17 17:51:50,769 - INFO - Evaluating PPO_custom_reward model...
2024-07-17 17:51:51,035 - INFO - Mean reward: 0.04 Std reward:0.19595917942265426
2024-07-17 17:51:51,035 - INFO - Testing PPO_custom_reward model...
2024-07-17 17:51:53,508 - INFO - PPO_custom_reward - Success rate: 0.02
2024-07-17 17:51:53,508 - INFO - PPO_custom_reward - Average reward: 0.02
```

After executing multiple times the evaluating and testing part, from the results, we observe the following key points:

Success Rate and Average Reward:
The PPO model has the highest success rate (in the above example is equal to 0.83) and average reward (0.83), making it the most successful in navigating the FrozenLake environment.
The DQN model follows closely with a success rate of 0.81 and an average reward of 0.81.
Custom policy models for both DQN and PPO show slightly lower success rates and average rewards compared to their default setups.
About the custom reward PPO model, the low mean reward (0.04) with the custom reward wrapper suggests that the agent is facing challenges in learning to navigate the environment effectively and find the final goal (G).
This might mean that the reward logic of penalizing every step the agent takes (-1 reward) and only providing a reward (+10) when the agent successfully reaches the goal, doesn't help the agent to learn effectively. 
Also, a success rate of 0.02 means that the agent completed the task in 2% of the evaluation episodes. 

Mean Reward and Standard Deviation:
The PPO model again shows a good performance with a mean reward of 0.80 in the above example and a standard deviation of 0.40, meaning more consistent performance.
The DQN model has a mean reward of 0.73, again the above execution, with a standard deviation of 0.44, meaning slightly higher variability in performance.

Conclusion:
Based on the evaluation metrics and the overall performance, the PPO model is recommended for the FrozenLake-v1 environment. The reasons are:
Higher success rate (the PPO model achieves the highest success rate, meaning it is more likely to successfully finish the game), higher average reward (the PPO model also achieves the highest average reward, suggesting better performance in terms of total rewards per total episodes), and better consistency.
While the DQN model also performs well, the PPO model's success rate, average reward, and consistency make it the best algorithm selection for this environment.
