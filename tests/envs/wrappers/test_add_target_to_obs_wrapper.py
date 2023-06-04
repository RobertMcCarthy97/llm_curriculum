import unittest
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from llm_curriculum.envs.wrappers import AddTargetToObsWrapper


class TestAddTargetToObsWrapper(unittest.TestCase):
    def setUp(self):
        # Set up a simple environment with two Box spaces
        obs_space = spaces.Dict(
            {
                "observation": spaces.Box(low=-1, high=1, shape=(4,), dtype=float),
                "desired_goal": spaces.Box(low=-1, high=1, shape=(3,), dtype=float),
            }
        )
        self.env = gym.Env()
        self.env.observation_space = obs_space
        self.wrapper = AddTargetToObsWrapper(self.env)

    def test_observation_space(self):
        # Make sure the observation space has the correct shape
        self.assertEqual((7,), self.wrapper.observation_space.shape)

    def test_observation(self):
        # Make sure the observation function returns the concatenated observation and goal arrays
        obs = {"observation": np.ones((4,)), "desired_goal": np.zeros((3,))}
        expected_obs = np.concatenate([np.ones((4,)), np.zeros((3,))], axis=0)
        new_obs = self.wrapper.observation(obs)
        self.assertTrue(np.array_equal(expected_obs, new_obs))
