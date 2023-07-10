import llm_curriculum.envs.minimal_minigrid.envs
import gymnasium as gym
import json
import re
import collections

from typing import List
from pathlib import Path
from minigrid.wrappers import FullyObsWrapper
from llm_curriculum.envs.minimal_minigrid.prompting.api import (
    chat_completion_request,
    get_default_system_message,
)
from llm_curriculum.envs.minimal_minigrid.description import (
    parse_agent,
    parse_field_of_view,
    describe_env,
)
from llm_curriculum.envs.minimal_minigrid.prompting.prompt import (
    load_prompt_template,
    parse_objectives,
    parse_function,
    parse_function_name,
)
from llm_curriculum.envs.minimal_minigrid.prompting.message import (
    save_messages,
    load_messages,
    save_obj,
    load_obj,
)

ENV_IDS = ["MiniGrid-IsNextTo-6x6-N2-v0", "MiniGrid-UnlockRed-v0"]

# Run decomposition 5x per environment
SAVE_DIR = Path(__file__).parent / "data"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def load_decompositions(save_dir=SAVE_DIR):
    decompositions = []
    prompt_type = "decomposition"
    for env_id in ENV_IDS:
        for trial in range(2):
            objectives_filepath = (
                SAVE_DIR / f"objectives_{prompt_type}_{env_id}_{trial}.json"
            )
            if not objectives_filepath.exists():
                print(f"Skipping {objectives_filepath} because it doesn't exist")
                continue
            env_decompositions = load_obj(objectives_filepath)
            decompositions.append((env_id, trial, env_decompositions))
    return decompositions


def validate_decomposition(d):
    env_id, trial, objectives = d
    assert isinstance(objectives, List)
    for objective in objectives:
        assert isinstance(objective, str)


def load_rewards(save_dir=SAVE_DIR):
    rewards = []
    prompt_type = "reward"
    for env_id in ENV_IDS:
        for trial in range(2):
            objectives_filepath = SAVE_DIR / f"objectives_decomposition_{env_id}_0.json"
            if not objectives_filepath.exists():
                print(f"Skipping {objectives_filepath} because it doesn't exist")
                continue
            objectives = load_obj(objectives_filepath)
            for objective in objectives:
                reward_function_filepath = (
                    SAVE_DIR
                    / f"reward_function_{prompt_type}_{env_id}_{objective}_{trial}.json"
                )
                if not reward_function_filepath.exists():
                    print(
                        f"Skipping {reward_function_filepath} because it doesn't exist"
                    )
                    continue
                reward_function = load_obj(reward_function_filepath)
                rewards.append((env_id, objective, trial, reward_function))
    return rewards


def validate_rewards(r):
    env_id, objective, trial, reward_function = r
    assert isinstance(objective, str)
    assert isinstance(reward_function, str)
    assert reward_function.strip().startswith("def ")

    # Check if function is well-formatted and callable
    function_name = parse_function_name(reward_function)
    exec(reward_function)
    function = locals()[function_name]
    assert callable(function)

    # Unit test on sample input
    create_obj = lambda: {"position": (0, 0)}
    sample_input = {
        "agent_info": {"position": (0, 0), "direction": 0, "carrying": "nothing"},
        # Defaultdict creates objects on the fly
        "field_of_view": collections.defaultdict(create_obj),
    }
    retval = function(sample_input)
    assert isinstance(retval, bool)


if __name__ == "__main__":
    decompositions = load_decompositions()
    for d in decompositions:
        validate_decomposition(d)
        print(d)

    rewards = load_rewards()
    for r in rewards:
        validate_rewards(r)
        env_id, objective, trial, reward_function = r
        print(" ****** ******")
        print("objective: ", objective)
        print(reward_function)
