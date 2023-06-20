import numpy as np
import gymnasium as gym


class AddExtraObjectsToObsWrapper(gym.ObservationWrapper):
    """
    Add the position of the red target to the observartion space
    """

    def __init__(self, env, add_target=True, add_drawer=False):
        super().__init__(env)
        self._env = env
        self.add_target = add_target
        self.add_drawer = add_drawer

        new_obs_size = env.observation_space["observation"].shape[0]
        if self.add_target:
            new_obs_size += env.observation_space["desired_goal"].shape[0]
        if self.add_drawer:
            new_obs_size += env.observation_space["drawer_state"].shape[0]

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_obs_size,)
        )

    def observation(self, observation):
        obs = observation["observation"]
        if self.add_target:
            goal = observation["desired_goal"]
            obs = np.concatenate([obs, goal], axis=0)
        if self.add_drawer:
            drawer_state = observation["drawer_state"]
            obs = np.concatenate([obs, drawer_state], axis=0)
        return obs
