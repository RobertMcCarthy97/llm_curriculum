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
)
from llm_curriculum.envs.minimal_minigrid.prompting.message import (
    save_messages,
    load_messages,
    save_obj,
    load_obj,
)

DECOMPOSITION_PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent / "decomposition_prompt_template.txt"
)
REWARD_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "reward_prompt_template.txt"


ENV_IDS = ["MiniGrid-IsNextTo-6x6-N2-v0", "MiniGrid-UnlockRed-v0"]

# Run decomposition 5x per environment
SAVE_DIR = Path(__file__).parent / "data"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def make_prompt(env_id: str) -> str:
    env = gym.make(env_id, render_mode="human")
    env = FullyObsWrapper(env)
    obs, _ = env.reset()

    env_str = describe_env(env)
    mission_str = obs["mission"]

    prompt_template = load_prompt_template(DECOMPOSITION_PROMPT_TEMPLATE_PATH)
    prompt = prompt_template.replace("${environment}", env_str).replace(
        "${mission}", mission_str
    )
    return prompt


def validate_response(response):
    response = response.json()
    assert "choices" in response

    # Ensure top reply is valid and complete
    assert response["choices"][0]["finish_reason"] == "stop"


def get_decomposition(decomposition_prompt):
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


def eval_decomposition(env_ids=ENV_IDS, n_trials=2):

    prompt_type = "decomposition"
    for env_id in env_ids:
        decomposition_prompt = make_prompt(env_id)
        print(" ****** DECOMPOSITION PROMPT ****** ")
        print(decomposition_prompt)

        for trial in range(n_trials):
            print(f" ****** TRIAL {trial} ****** ")
            messages, objectives = get_decomposition(decomposition_prompt)
            print(f"Objectives: ")
            print(objectives)

            # Save messages
            message_filepath = (
                SAVE_DIR / f"messages_{prompt_type}_{env_id}_{trial}.json"
            )
            save_messages(messages, message_filepath)
            print(f"Saved messages to {message_filepath}")

            # Save objectives
            obj_filepath = SAVE_DIR / f"objectives_{prompt_type}_{env_id}_{trial}.json"
            save_obj(objectives, SAVE_DIR / obj_filepath)
            print(f"Saved objectives to {SAVE_DIR / obj_filepath}")


if __name__ == "__main__":
    eval_decomposition()
    # eval_reward()
