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

DECOMPOSITION_PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent / "decomposition_prompt_template.txt"
)
REWARD_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "reward_prompt_template.txt"


def load_prompt_template(path: Path):
    with open(path) as f:
        return f.read()


def is_objective(string: str) -> bool:
    pattern = r"^\d+\.\s"  # This regular expression pattern matches one or more digits, followed by '. ', at the start of a string.
    return bool(re.match(pattern, string))


def parse_objective_tag(string):
    pattern = "<objectives>(.*?)</objectives>"
    match = re.search(pattern, string, re.DOTALL)
    if match:
        return match.group(1)
    return None


def parse_objectives(reply: str) -> List[str]:
    objectives = []
    reply = parse_objective_tag(reply)
    lines = reply.split("\n")
    pattern = r"^\d+\.\s"  # This regular expression pattern matches one or more digits, followed by '. ', at the start of a string.

    for line in lines:
        match = re.match(pattern, line)
        if match is not None:
            start, end = match.span()
            match_len = end - start
            objectives.append(line[match_len:])
    return objectives


if __name__ == "__main__":

    env = gym.make("MiniGrid-IsNextTo-6x6-N2-v0", render_mode="human")
    env = FullyObsWrapper(env)
    obs, _ = env.reset()

    env_str = describe_env(env)
    mission_str = obs["mission"]

    print(" ****** DECOMPOSITION PROMPT ****** ")
    prompt_template = load_prompt_template(DECOMPOSITION_PROMPT_TEMPLATE_PATH)
    prompt = prompt_template.replace("${environment}", env_str).replace(
        "${mission}", mission_str
    )

    print(prompt)

    # Send the request
    print(" ****** DECOMPOSITION RESPONSE ****** ")
    messages = [
        {"role": "system", "content": get_default_system_message()},
        {"role": "user", "content": prompt},
    ]
    response = chat_completion_request(messages)
    assistant_message = response.json()["choices"][0]["message"]
    # Add response to history
    messages.append(assistant_message)

    reply = assistant_message["content"]
    print(reply)

    # Parse the response
    print(" ****** OBJECTIVES ****** ")
    objectives = parse_objectives(reply)
    print(objectives)

    # Define reward coding prompt
    print(" ****** REWARD PROMPT ****** ")
    reward_prompt_template = load_prompt_template(REWARD_PROMPT_TEMPLATE_PATH)
    reward_prompt = reward_prompt_template.replace("${state}", objectives[0])
    print(reward_prompt)

    # Send the request
    print(" ****** REWARD RESPONSE ****** ")
    messages.append({"role": "user", "content": reward_prompt_template})
    response = chat_completion_request(messages)
    assistant_message = response.json()["choices"][0]["message"]
    # Add response to history
    messages.append(assistant_message)

    reply = assistant_message["content"]
    print(reply)
