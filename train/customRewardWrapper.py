from gymnasium import RewardWrapper

class CustomRewardWrapper(RewardWrapper):
    """
    A custom reward wrapper for modifying the reward logic of the environment.
    By overriding the reward method, CustomRewardWrapper can modify the reward logic
    of the base environment (env) without altering the original code of RewardWrapper.

    This custom reward wrapper assigns a penalty of -1 for each step taken and a reward
    of 10 forn reaching the goal.

    Args:
    - env: The base FrozenLake-v1 environment to wrap.

    Methods:
    - reward(reward: float) -> float:
        Overrides the reward method to modify the reward logic.

    Returns:
        int: reward

    """
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == 0:
            return -1
        elif reward == 1:
            return 10
        return reward