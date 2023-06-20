import unittest
import gym
import numpy as np  # noqa
from gym import spaces
from llm_curriculum.envs.agent_conductor import AgentConductor
from llm_curriculum.envs.wrappers import CurriculumEnvWrapper


class DummyEnv(gym.Env):
    """
    Simple dummy environment with a Box observation space
    and discrete action space.
    """

    def __init__(self):
        obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
        act_space = spaces.Discrete(2)
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        obs = self.reset()
        reward = 0
        done = False
        info = {}
        return obs, reward, done, info


class TestCurriculumEnvWrapper(unittest.TestCase):
    def setUp(self):
        # Set up a simple environment with a Box observation space
        env = DummyEnv()
        # Note: AgentConductor is currently hard-coded to Fetch environment
        # So this test suite isn't going to work with Dummy env.
        ac = AgentConductor(env, high_level_task_names=["foo", "bar"])
        max_ep_len = 50
        self.wrapper = CurriculumEnvWrapper(env, ac, max_ep_len=max_ep_len)


"""
    def test_reset(self):
        # Make sure resetting the environment returns an observation and an info dictionary
        obs, info = self.wrapper.reset()
        self.assertIsInstance(obs, dict)
        self.assertIn("observation", obs)
        self.assertIn("desired_goal", obs)
        self.assertIsInstance(obs["observation"], np.ndarray)
        self.assertIsInstance(obs["desired_goal"], np.ndarray)
        self.assertEqual((4,), obs["observation"].shape)
        self.assertEqual((2,), obs["desired_goal"].shape)
        self.assertIsInstance(info, dict)

    def test_step(self):
        # Make sure stepping the environment returns an observation, reward, done flag, and info dictionary
        obs, _, done, info = self.wrapper.step(0)
        self.assertIsInstance(obs, dict)
        self.assertIn("observation", obs)
        self.assertIn("desired_goal", obs)
        self.assertIsInstance(obs["observation"], np.ndarray)
        self.assertIsInstance(obs["desired_goal"], np.ndarray)
        self.assertEqual((4,), obs["observation"].shape)
        self.assertEqual((2,), obs["desired_goal"].shape)
        self.assertFalse(done)
        self.assertIsInstance(info, dict)

    def test_init_obs_space(self):
        # Make sure the observation space is set up correctly in the wrapper
        expected_shape = (6,)  # 4 for observation + 2 for desired goal
        self.assertIsInstance(self.wrapper.observation_space, spaces.Dict)
        self.assertIn("observation", self.wrapper.observation_space.spaces)
        self.assertIn("desired_goal", self.wrapper.observation_space.spaces)
        self.assertIsInstance(self.wrapper.observation_space.spaces["observation"], spaces.Box)
        self.assertIsInstance(self.wrapper.observation_space.spaces["desired_goal"], spaces.Box)
        self.assertEqual(expected_shape, self.wrapper.observation_space.shape)
    
"""
