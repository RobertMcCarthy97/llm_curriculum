import minigrid  # noqa
import gymnasium as gym

from typing import List
from llm_curriculum.envs.minigrid.tasks import (
    BaseTask,
    GoToObjectTask,
    PickUpObjectTask,
    OpenDoorTask,
)
from llm_curriculum.envs.minigrid.grid_utils import ObjectDescription
from llm_curriculum.envs.minigrid.env_wrapper import env_to_str, MinigridTaskEnvWrapper
from llm_curriculum.envs.minigrid.prompting.api import (
    chat_completion_request,
    pretty_print_conversation,
)


def make_prompt(env):
    return (
        f"""
You are controlling a simulated agent in a 2D grid to complete tasks. 
Describe a sequence of intermediate objectives to complete the overall mission. 
---
{env_to_str(env)}
---"""
        + """
Follow this template:

[Thought:  ${description of reasoning process}]
[repeat above any number of times needed...]

The objectives are:
<start of description>
[#. ${short description of objective}]
[repeat above any number of times needed...]
<end of description>
---
"""
        + """
Rules: 
1. Each objective should follow the template: ${verb} the ${object}
2. The allowed verbs are: "go to", "pick up", "put down", "open", "close"
3. The allowed objects are: "box", "key", "door", "ball", "goal", "wall", "lava"
4. Objects should be described only in terms of their types. 
5. Do not include any information relating to color or coordinate. 
"""
    )


def parse_response(response: "Response") -> str:
    return response.json()["choices"][0]["text"]["content"]


def parse_task_descriptions(content: str) -> List[str]:
    """Parse the chatbot reply into a list of task descriptions"""
    tasks = []
    for line in content.split("\n"):
        if len(line) > 0 and line[0].isnumeric():
            line = line[3:]
            tasks.append(line)
    return tasks


def make_tasks(task_descriptions: List[str]) -> List[BaseTask]:

    tasks = []
    for task_desc in task_descriptions:
        task_desc = task_desc.lower()
        if "go to" in task_desc:
            task_class = GoToObjectTask
        elif "pick up" in task_desc:
            task_class = PickUpObjectTask
        elif "open" in task_desc:
            task_class = OpenDoorTask
        else:
            raise ValueError(f"Unknown verb in task description: {task_desc}")

        if "key" in task_desc:
            object = "key"
        elif "door" in task_desc:
            object = "door"
        elif "box" in task_desc:
            object = "box"
        else:
            raise ValueError(f"Unknown object in task description: {task_desc}")

        object_desc = ObjectDescription(type=object, color="any")
        tasks.append(task_class(object_desc))

    return tasks


if __name__ == "__main__":
    env = gym.make("MiniGrid-UnlockPickup-v0", render_mode="human")
    env.reset()
    env_str = env_to_str(env)

    messages = [
        {
            "role": "system",
            "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture. Knowledge cutoff: 2021-09. Current date: 2023-05-04. ",
        },
        {"role": "user", "content": make_prompt(env)},
    ]

    # Note: This is equivalent to one-shot completion
    # TODO: Can we ask for multiple completions and then ask GPT to select the best one?

    response = chat_completion_request(messages)
    assistant_message = response.json()["choices"][0]["message"]
    messages.append(assistant_message)

    response_content = assistant_message["content"]

    task_descs = parse_task_descriptions(response_content)
    tasks = make_tasks(task_descs)
    print(tasks)
    make_task_fn = lambda env: tasks

    env = MinigridTaskEnvWrapper(env, make_task_fn)
    obs, _ = env.reset()

    from llm_curriculum.envs.minigrid.manual_control import ManualControl

    controller = ManualControl(env)
    controller.start()
