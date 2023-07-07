import llm_curriculum.envs.minimal_minigrid.envs
import gymnasium as gym
import json

from pathlib import Path
from minigrid.wrappers import FullyObsWrapper
from llm_curriculum.envs.minimal_minigrid.prompting.api import (
    chat_completion_request,
    get_default_system_message,
)
from llm_curriculum.envs.minimal_minigrid.description import (
    parse_agent,
    parse_field_of_view,
)

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "decomposition_prompt_template.txt"


def load_prompt_template(path: Path):
    with open(path) as f:
        return f.read()


if __name__ == "__main__":

    env = gym.make("MiniGrid-IsNextTo-6x6-N2-v0", render_mode="human")
    env = FullyObsWrapper(env)
    obs, _ = env.reset()

    llm_obs = {
        "agent_info": parse_agent(env),
        "field_of_view": parse_field_of_view(obs["image"]),
    }
    print(llm_obs)

    llm_obs_str = json.dumps(llm_obs)
    mission_str = obs["mission"]

    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    prompt = prompt_template.replace("${environment}", llm_obs_str).replace(
        "${mission}", mission_str
    )

    print(prompt)

    # Send the request
    messages = [
        {"role": "system", "content": get_default_system_message()},
        {"role": "user", "content": prompt},
    ]
    response = chat_completion_request(messages)
    assistant_message = response.json()["choices"][0]["message"]
    reply = assistant_message["content"]
    print(reply)

    # TODO: Parse the response
