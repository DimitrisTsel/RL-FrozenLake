import logging
import requests
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API endpoint URLs
API_URL = 'http://localhost:5005'
NEW_GAME_URL = f'{API_URL}/new_game'
RESET_URL = f'{API_URL}/reset'
STEP_URL = f'{API_URL}/step'

def reset_env(game_id):
    """
    Reset the environment for the specified game.

    Args:
        game_id (str): The ID of the game to reset.

    Returns:
        np.ndarray: The initial observation after reset.
    """
    response = requests.post(RESET_URL, json={'game_id': game_id}, timeout=10)
    data = response.json()
    if data['success']:
        observation = data['observation'][0]
        return observation
    else:
        return None

def step_env(action, game_id):
    """
    Take a step in the environment for the specified game.

    Args:
        action (int): The action to perform.
        game_id (str): The uuid of the game.

    Returns:
        tuple: The next observation, reward, done flag, and info dictionary.
    """
    response = requests.post(STEP_URL,
                             json={'game_id': game_id, 'action': int(action)}, timeout=10)
    data = response.json()
    if data['success']:
        next_observation = np.array(data['observation'])
        reward = data['reward']
        done = data['done']
        info = data['info']
        return next_observation, reward, done, info
    else:
        return None

def new_game():
    """
    Start a new game.

    Returns:
        The game uuid.
    """
    response = requests.get(NEW_GAME_URL, timeout=10)
    data = response.json()
    if data['success']:
        game_id = data['game_id']
        return game_id
    else:

        raise Exception('An error occured ducing the env creation!')

def evaluate_model(model, env, n_eval_episodes=100):
    """
    Evaluate the model using stable-baselines3's evaluate_policy.

    Args:
        model: The model to evaluate.
        env: The environment to evaluate the model on.
        n_eval_episodes (int): The number of episodes to evaluate.

    Returns:
        tuple: Mean reward and standard deviation of the reward.
    """
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    logger.info(f"Mean reward: {mean_reward} Std reward:{std_reward}")
    return mean_reward, std_reward

def test_model(model, model_name, n_episodes=100):
    """
    Test the model by running multiple episodes and calculating success rate.

    Args:
        model: The model to test.
        model_name (str): The name of the model.
        n_episodes (int): The number of episodes to run.

    Returns:
        tuple: Success rate and average reward.
    """
    success_count = 0
    total_reward = 0
    success_rate = 0
    average_reward = 0
    for _ in range(n_episodes):
        game_id = new_game()
        observation = reset_env(game_id)
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, _ = step_env(action, game_id)
            total_reward += reward
            if done and reward == 1:  # Success condition for FrozenLake
                success_count += 1

    success_rate = success_count / n_episodes
    average_reward = total_reward / n_episodes
    logger.info("%s - Success rate: %s", model_name, success_rate)
    logger.info("%s - Average reward: %s", model_name, average_reward)
    return success_rate, average_reward

def main():
    """
    Main function to load models, evaluate, and test them.

    """
    env = gym.make("FrozenLake-v1", map_name='4x4', is_slippery=True)
    env = Monitor(env)

    # Load best models
    dqn_model_path = "models/DQN/280000.zip"
    dqn_custom_policy_model_path = "models/DQN_policy_kwargs/280000.zip"
    ppo_model_path = "models/PPO/130000.zip"
    ppo_custom_policy_model_path = "models/PPO_policy_kwargs/250000.zip"

    dqn_model = DQN.load(dqn_model_path, env=env)
    dqn_custom_policy_model = DQN.load(dqn_custom_policy_model_path, env=env)
    ppo_model = PPO.load(ppo_model_path, env=env)
    ppo_custom_policy_model = PPO.load(ppo_custom_policy_model_path, env=env)


    logger.info("Evaluating DQN model...")
    evaluate_model(dqn_model, env)

    logger.info("Testing DQN model...")
    test_model(dqn_model, "DQN")

    logger.info("Evaluating DQN_custom_policy model...")
    evaluate_model(dqn_custom_policy_model, env)

    logger.info("Testing DQN_custom_policy model...")
    test_model(dqn_custom_policy_model, "DQN_custom_policy")

    logger.info("Evaluating PPO model...")
    evaluate_model(ppo_model, env)

    logger.info("Testing PPO model...")
    test_model(ppo_model, "PPO")

    logger.info("Evaluating PPO_custom_policy model...")
    evaluate_model(ppo_custom_policy_model, env)

    logger.info("Testing PPO_custom_policy model...")
    test_model(ppo_custom_policy_model, "PPO_custom_policy")



if __name__ == "__main__":
    main()
