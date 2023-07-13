import llm_curriculum.envs.minimal_minigrid.envs
import gymnasium as gym
import json
import re

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
)
from llm_curriculum.envs.minimal_minigrid.prompting.message import (
    save_messages,
    load_messages,
    save_obj,
    load_obj,
)
from copy import deepcopy

DECOMPOSITION_PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent / "decomposition_prompt_template.txt"
)
REWARD_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "reward_prompt_template.txt"


ENV_IDS = ["MiniGrid-UnlockPickupFixed-6x6-v0"]

# Run decomposition 5x per environment
SAVE_DIR = Path(__file__).parent / "data"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def make_decomposition_prompt(env_id: str) -> str:
    env = gym.make(env_id, render_mode="human")
    obs, _ = env.reset()

    env_str = describe_env(env.unwrapped)
    mission_str = obs["mission"]

    prompt_template = load_prompt_template(DECOMPOSITION_PROMPT_TEMPLATE_PATH)
    prompt = prompt_template.replace("${environment}", env_str).replace(
        "${mission}", mission_str
    )
    return prompt


def make_reward_prompt(objective) -> str:
    prompt_template = load_prompt_template(REWARD_PROMPT_TEMPLATE_PATH)
    prompt = prompt_template.replace("${state}", objective)
    return prompt


def validate_response(response):
    response = response.json()
    assert "choices" in response

    # Ensure top reply is valid and complete
    assert response["choices"][0]["finish_reason"] == "stop"


def send_message(messages: List[dict]):
    response = chat_completion_request(messages)
    validate_response(response)
    assistant_message = response.json()["choices"][0]["message"]
    # Add response to history
    messages.append(assistant_message)
    reply = assistant_message["content"]
    return messages, reply


def get_objectives(decomposition_prompt):
    messages = [
        {"role": "system", "content": get_default_system_message()},
        {"role": "user", "content": decomposition_prompt},
    ]
    response = chat_completion_request(messages)
    validate_response(response)

    assistant_message = response.json()["choices"][0]["message"]
    # Add response to history
    messages.append(assistant_message)
    reply = assistant_message["content"]
    objectives = parse_objectives(reply)
    return messages, objectives


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


def get_decomposition(env_ids=ENV_IDS, n_trials=2):

    prompt_type = "decomposition"
    for env_id in env_ids:
        decomposition_prompt = make_decomposition_prompt(env_id)
        env_type = env_id.split("-")[1]
        print(" ****** DECOMPOSITION PROMPT ****** ")
        print(decomposition_prompt)

        for trial in range(n_trials):
            print(f" ****** TRIAL {trial} ****** ")
            messages, objectives = get_objectives(decomposition_prompt)
            print(f"Objectives: ")
            print(objectives)

            # Save messages
            env_save_dir = SAVE_DIR / camel_to_snake(env_type)
            env_save_dir.mkdir(parents=True, exist_ok=True)
            message_filepath = (
                env_save_dir / f"messages_{prompt_type}_{env_type}_{trial}.json"
            )
            save_messages(messages, message_filepath)
            print(f"Saved messages to {message_filepath}")

            # Save objectives
            obj_filepath = (
                env_save_dir / f"objectives_{prompt_type}_{env_type}_{trial}.json"
            )
            save_obj(objectives, SAVE_DIR / obj_filepath)
            print(f"Saved objectives to {SAVE_DIR / obj_filepath}")


def get_reward(env_ids=ENV_IDS, n_trials=2):
    prompt_type = "reward"
    for env_id in env_ids:
        env_type = env_id.split("-")[1]
        env_save_dir = SAVE_DIR / camel_to_snake(env_type)
        env_save_dir.mkdir(parents=True, exist_ok=True)
        decomposition_messages = load_messages(
            env_save_dir / f"messages_decomposition_{env_type}_0.json"
        )
        decomposition_objectives = load_obj(
            env_save_dir / f"objectives_decomposition_{env_type}_0.json"
        )
        for objective in decomposition_objectives:
            reward_prompt = make_reward_prompt(objective)
            print(" ****** REWARD PROMPT ****** ")
            print(reward_prompt)
            for trial in range(n_trials):
                print(f" ****** TRIAL {trial} ****** ")
                reward_messages = deepcopy(decomposition_messages)
                reward_messages.append({"role": "user", "content": reward_prompt})
                messages, reply = send_message(reward_messages)

                print(f"Reply: ")
                print(reply)
                print()
                reward_function = parse_function(reply)
                print(f"Reward function: ")
                print(reward_function)

                # Save messages
                message_filepath = (
                    env_save_dir
                    / f"messages_{prompt_type}_{env_type}_{objective}_{trial}.json"
                )
                save_messages(messages, message_filepath)
                print(f"Saved messages to {message_filepath}")

                # Save reward function
                reward_function_filepath = (
                    env_save_dir
                    / f"reward_function_{prompt_type}_{env_type}_{objective}_{trial}.json"
                )
                save_obj(reward_function, reward_function_filepath)
                print(f"Saved reward function to {reward_function_filepath}")


if __name__ == "__main__":
    get_decomposition()
    get_reward()
