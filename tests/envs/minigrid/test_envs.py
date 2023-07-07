import pytest
import gymnasium as gym
import llm_curriculum.envs.minigrid


@pytest.mark.parametrize(
    "env_id",
    [
        "MiniGrid-UnlockPickupDecomposed-v0",
        # Require API calls
        # "MiniGrid-UnlockDecomposedAutomated-v1",
        # "MiniGrid-UnlockPickupDecomposedAutomated-v0",
        # "MiniGrid-BlockedUnlockPickupDecomposedAutomated-v1",
    ],
)
def test_env(env_id):
    env = gym.make(env_id)
