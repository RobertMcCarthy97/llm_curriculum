import numpy as np
import gymnasium as gym
import gym as gym_old


class MTEnvWrapper(gym_old.Wrapper):
    """
    A multi-task environment wrapper that modifies the observation space of an underlying Gym environment.

    This wrapper modifies the observation space of the underlying environment by adding a task observation
    component to the original observation. The task observation component represents the task index.

    Args:
        env (gym.Env): The underlying Gym environment to be wrapped.
        task_index (int): The task index associated with this instance of the environment wrapper.

    Attributes:
        observation_space (gym.spaces.Dict): The modified observation space of the environment. It is a dictionary
            that contains two keys:
                - 'env_obs': The original observation space of the underlying environment.
                - 'task_obs': A Box space representing the task index.

    Methods:
        reset(): Resets the environment and returns the modified observation.
        step(action): Performs a step in the environment and returns the modified observation, reward, done, and info.

    """

    def __init__(self, env, task_index: int):
        super().__init__(env)
        self._env = env
        self._task_index = task_index

        self.observation_space = gym_old.spaces.Dict(
            {
                "env_obs": env.observation_space["observation"],
                "task_obs": gym_old.spaces.Box(low=0, high=np.inf, shape=(1,)),
            }
        )

    def reset(self):
        """
        Resets the environment and returns the modified observation.

        Returns:
            observation (dict): The modified observation containing both the environment observation and the
                task observation component.

        """
        obs = self._env.reset()
        return self.mod_observation(obs)

    def step(self, action):
        """
        Performs a step in the environment and returns the modified observation, reward, done, and info.

        Args:
            action: The action to take in the environment.

        Returns:
            observation (dict): The modified observation containing both the environment observation and the
                task observation component.
            reward (float): The reward obtained from the action.
            done (bool): Indicates whether the episode is finished.
            info (dict): Additional information about the step.

        """
        obs, reward, done, info = self._env.step(action)
        return self.mod_observation(obs), reward, done, info

    def mod_observation(self, observation):
        """
        Modifies the observation by adding the task observation component.

        Args:
            observation: The original observation obtained from the underlying environment.

        Returns:
            modified_observation (dict): The modified observation containing both the environment observation and the
                task observation component.

        """
        task_obs = np.array([self._task_index])
        return {"env_obs": observation["observation"], "task_obs": task_obs}
