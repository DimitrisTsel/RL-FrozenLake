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
│ ├── model_train_DQN.py
│ └── model_train_PPO.py
└── requirements.txt
```

- `logs/`: Directory to store TensorBoard logs.
- `models/`: Directory to save trained models.
- `src/`: Source code for the API and client.
  - `client.py`: Script to evaluate and test models using the provided API.
  - `FrozenLakeAPI_structure.py`: Flask application to create and interact with FrozenLake environments.
- `train/`: Scripts to train DQN and PPO models.
  - `model_train_DQN.py`: Script to train a DQN model.
  - `model_train_PPO.py`: Script to train a PPO model.

## Getting Started

### Environment Setup

- Python 3.8+
- create an isolated python virtual environment
  - ` python -m venv venv`
  - `source venv/bin/activate`
- Install dependencies using `pip`:


  `pip install -r requirements.txt`

### Train the models
#### DQN model
To train a DQN model, run: `python train/model_train_DQN.py`

This will save the trained model in the models/DQN directory and log training details in the logs directory.

#### PPO model
To train a PPO model, run: `python train/model_train_PPO.py`

### Running the API
To start the FrozenLake API, run: `python src/FrozenLakeAPI_structure.py`

The server host adress is  http://localhost:5005.

### API Endpoints
- [GET]  /new_game: Creates a new game and returns a game_id.
- [POST] /reset: Resets the environment for the specified game_id.
- [POST] /step: Takes a step in the environment for the specified game_id using the provided action.

#### Evaluating and Testing Models
To evaluate and test the models, run: `python src/client.py`
This script will:
- Load pre-trained models.
- Evaluate each model using evaluate_policy.
- Test each model by running multiple episodes and calculating success rate and average reward.
