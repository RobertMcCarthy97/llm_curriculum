import pytest
import gymnasium as gym
import llm_curriculum.envs.minimal_minigrid.envs
from minigrid.wrappers import FlatObsWrapper


@pytest.mark.parametrize(
    "env_id",
    [
        "MiniGrid-IsNextTo-6x6-v0",
        "MiniGrid-IsNextTo-6x6-DecomposedReward-v0",
        "MiniGrid-IsNextTo-12x12-v0",
        "MiniGrid-IsNextTo-12x12-DecomposedReward-v0",
        "MiniGrid-UnlockRed-6x6-v0",
        "MiniGrid-UnlockRed-6x6-DecomposedReward-v0",
        "MiniGrid-UnlockRed-12x12-v0",
        "MiniGrid-UnlockRed-12x12-DecomposedReward-v0",
    ],
)
def test_env(env_id):
    env = gym.make(env_id)
    env = FlatObsWrapper(env)
    obs, info = env.reset()

    for trial in range(5):
        obs, info = env.reset()
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break


@pytest.mark.parametrize(
    "env_id",
    [
        "MiniGrid-IsNextTo-6x6-v0",
        "MiniGrid-IsNextTo-6x6-DecomposedReward-v0",
        "MiniGrid-IsNextTo-12x12-v0",
        "MiniGrid-IsNextTo-12x12-DecomposedReward-v0",
        "MiniGrid-UnlockRed-6x6-v0",
        "MiniGrid-UnlockRed-6x6-DecomposedReward-v0",
        "MiniGrid-UnlockRed-12x12-v0",
        "MiniGrid-UnlockRed-12x12-DecomposedReward-v0",
    ],
)
def test_env_image_shape(env_id):
    env = gym.make(env_id)
    assert env.observation_space["image"].shape == (7, 7, 3)
