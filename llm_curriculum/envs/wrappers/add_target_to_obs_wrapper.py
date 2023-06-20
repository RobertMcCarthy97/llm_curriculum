import numpy as np
import gymnasium as gym


class AddTargetToObsWrapper(gym.ObservationWrapper):
    """
    Add the position of the red target to the observartion space
    """

    def __init__(self, env):
        super().__init__(env)
        self._env = env

        new_obs_size = (
            env.observation_space["observation"].shape[0]
            + env.observation_space["desired_goal"].shape[0]
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_obs_size,)
        )

    def observation(self, observation):
        obs = observation["observation"]
        goal = observation["desired_goal"]
        return np.concatenate([obs, goal], axis=0)
