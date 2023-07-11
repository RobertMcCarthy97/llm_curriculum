import gymnasium as gym
import pathlib
from gymnasium import register

from minigrid.wrappers import FullyObsWrapper
from llm_curriculum.envs.minimal_minigrid.envs.wrappers import (
    DecomposedRewardWrapper,
    FullyObsInfoWrapper,
)
from llm_curriculum.envs.minimal_minigrid.prompting.prompt import parse_function_name
from llm_curriculum.envs.minimal_minigrid.prompting.message import (
    load_obj,
)

register(
    "MiniGrid-IsNextTo-6x6-v0",
    entry_point="llm_curriculum.envs.minimal_minigrid.envs.is_next_to:IsNextToEnv",
)

register(
    "MiniGrid-IsNextTo-12x12-v0",
    entry_point="llm_curriculum.envs.minimal_minigrid.envs.is_next_to:IsNextToEnv",
    kwargs={"size": 12},
)

register(
    "MiniGrid-UnlockRed-6x6-v0",
    entry_point="llm_curriculum.envs.minimal_minigrid.envs.unlock:UnlockEnv",
)

register(
    "MiniGrid-UnlockRed-12x12-v0",
    entry_point="llm_curriculum.envs.minimal_minigrid.envs.unlock:UnlockEnv",
    kwargs={"room_size": 12},
)

register(
    "MiniGrid-UnlockPickup-6x6-v0",
    entry_point="llm_curriculum.envs.minimal_minigrid.envs.unlock_pickup:UnlockPickupEnv",
)

register(
    "MiniGrid-UnlockPickup-12x12-v0",
    entry_point="llm_curriculum.envs.minimal_minigrid.envs.unlock_pickup:UnlockPickupEnv",
    kwargs={"room_size": 12},
)


def make_decomposed_reward_env(
    env_id,
    objectives,
    reward_functions,
    enable_mission: bool,
    enable_reward: bool,
    **kwargs,
):
    env = gym.make(env_id, **kwargs)
    env = FullyObsInfoWrapper(env)
    env = DecomposedRewardWrapper(
        env, objectives, reward_functions, enable_mission, enable_reward
    )
    return env


root_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent
data_dir = root_dir / "llm_curriculum/envs/minimal_minigrid/prompting/data"

env_ids = [
    "MiniGrid-IsNextTo-6x6-v0",
    "MiniGrid-IsNextTo-12x12-v0",
    "MiniGrid-UnlockRed-6x6-v0",
    "MiniGrid-UnlockRed-12x12-v0",
    "MiniGrid-UnlockPickup-6x6-v0",
    "MiniGrid-UnlockPickup-12x12-v0",
]


def camel_to_snake(camelcase_string):
    snakecase_string = ""
    for char in camelcase_string:
        if char.isupper():
            snakecase_string += "_" + char.lower()
        else:
            snakecase_string += char
    if snakecase_string.startswith("_"):
        snakecase_string = snakecase_string[1:]
    return snakecase_string


for orig_env_id in env_ids:
    for enable_mission in [True, False]:
        for enable_reward in [True, False]:

            def make_env_factory(orig_env_id, enable_mission, enable_reward):
                def make_env(**kwargs):
                    env_type = orig_env_id.split("-")[1]
                    env_dir = data_dir / camel_to_snake(env_type)

                    objectives = load_obj(
                        env_dir / f"objectives_decomposition_{env_type}_0.json"
                    )
                    reward_functions = []
                    for objective in objectives:
                        rew_fn_str = load_obj(
                            env_dir
                            / f"reward_function_reward_{env_type}_{objective}_0.json"
                        )
                        rew_fn_name = parse_function_name(rew_fn_str)
                        exec(rew_fn_str)
                        rew_fn = locals()[rew_fn_name]
                        reward_functions.append(rew_fn)

                    def preprocess_objective(objective):
                        objective = (
                            objective.replace(".", "")
                            .replace("'", "")
                            .replace("_", " ")
                        )
                        return objective

                    objectives = [preprocess_objective(o) for o in objectives]

                    return make_decomposed_reward_env(
                        orig_env_id,
                        objectives,
                        reward_functions,
                        enable_mission,
                        enable_reward,
                        **kwargs,
                    )

                return make_env

            enable_mission_str = "" if enable_mission else "NoMission-"
            enable_reward_str = "" if enable_reward else "NoReward-"
            new_env_id = f"{orig_env_id[:-3]}-DecomposedReward-{enable_mission_str}{enable_reward_str}v0"
            register(
                new_env_id,
                entry_point=make_env_factory(
                    orig_env_id, enable_mission, enable_reward
                ),
            )
