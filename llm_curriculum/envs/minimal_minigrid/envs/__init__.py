import gymnasium as gym
import pathlib
from gymnasium import register

from minigrid.wrappers import FullyObsWrapper
from llm_curriculum.envs.minimal_minigrid.envs.wrappers import DecomposedRewardWrapper
from llm_curriculum.envs.minimal_minigrid.prompting.message import (
    load_obj,
)

register(
    "MiniGrid-IsNextTo-6x6-N2-v0",
    entry_point="llm_curriculum.envs.minimal_minigrid.envs.is_next_to:IsNextToEnv",
)

register(
    "MiniGrid-UnlockRed-v0",
    entry_point="llm_curriculum.envs.minimal_minigrid.envs.unlock:UnlockEnv",
)


def make_decomposed_reward_env(env_id, objectives, reward_functions, **kwargs):
    env = gym.make(env_id, **kwargs)
    env = FullyObsWrapper(env)
    env = DecomposedRewardWrapper(env, objectives, reward_functions)
    return env


root_dir = pathlib.Path(__file__).parent.parent.parent
data_dir = root_dir / "llm_curriculum/envs/minimal_minigrid/prompting/data"

env_prefixes = ["MiniGrid-IsNextTo-6x6-N2", "MiniGrid-UnlockRed"]
for env_prefix in env_prefixes:
    orig_env_id = f"{env_prefix}-v0"
    objectives = load_obj(data_dir / f"objectives_decomposition_{orig_env_id}_0.json")
    reward_functions = load_obj(
        data_dir / f"reward_function_reward_{orig_env_id}_0.json"
    )
    make_env_fn = lambda: make_decomposed_reward_env(
        orig_env_id, objectives, reward_functions
    )

    register(
        f"{env_prefix}-DecomposedReward-v0",
        entry_point=make_env_fn,
    )
