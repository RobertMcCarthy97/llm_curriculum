import pytest
import gymnasium as gym
import llm_curriculum.envs.minimal_minigrid.envs


@pytest.mark.parametrize(
    "env_id",
    [
        "MiniGrid-IsNextTo-6x6-N2-v0",
        "MiniGrid-UnlockRed-v0",
    ],
)
def test_env(env_id):
    env = gym.make(env_id)
    obs, info = env.reset()
