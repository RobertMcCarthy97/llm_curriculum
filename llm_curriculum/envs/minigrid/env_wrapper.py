import gymnasium as gym
import numpy as np

from typing import List, Callable

from minigrid.wrappers import ReseedWrapper
from llm_curriculum.envs.minigrid.tasks import BaseTask
from llm_curriculum.envs.minigrid.grid_utils import (
    get_object_pos,
    ObjectDescription,
    grid_to_str,
)
from llm_curriculum.envs.minigrid.tasks import (
    GoToObjectTask,
    PickUpObjectTask,
    OpenDoorTask,
)


def env_to_str(env: gym.Env):
    grid_str = grid_to_str(env.grid)
    agent_pos = env.agent_pos
    agent_dir = env.agent_dir
    agent_dir_str = ["right", "down", "left", "up"][agent_dir]
    env_str = (
        f"The environment consists of:\n"
        + "\n".join([("--" + line) for line in grid_str.split("\n ")])
        + f"The overall mission is: {env.mission}\n"
        f"The agent is at: {agent_pos}\n"
        f"The agent is facing: {agent_dir_str}"
    )
    return env_str


class MinigridTaskEnvWrapper(gym.Wrapper):
    """Wrap a Minigrid environment with a sequence of tasks

    Replace 'mission' with subtask
    Replace reward with subtask reward
    """

    def __init__(
        self, env: gym.Env, make_tasks_fn: Callable[[gym.Env], List[BaseTask]]
    ):
        super().__init__(env)
        self.make_tasks_fn = make_tasks_fn

    def get_current_task(self):
        return self.tasks[self.current_task_idx]

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.tasks = self.make_tasks_fn(self.env)
        self.current_task_idx = 0
        info["overall_mission"] = obs["mission"]
        obs["mission"] = self.get_current_task().to_string()
        return obs, info

    def step(self, action):
        obs, _, _, truncated, info = self.env.step(action)

        terminated = False
        task = self.get_current_task()
        task_success = task.check_success(self.env)
        if task_success:
            reward = 1
            if self.current_task_idx < len(self.tasks) - 1:
                self.current_task_idx += 1
            else:
                terminated = True
        else:
            reward = 0

        info["overall_mission"] = obs["mission"]
        obs["mission"] = self.get_current_task().to_string()

        return obs, reward, terminated, truncated, info


def make_automated_env(*args, **kwargs):

    from llm_curriculum.envs.minigrid.prompting.prompt import (
        make_prompt,
        parse_task_descriptions,
        chat_completion_request,
        make_tasks,
    )

    env = gym.make("MiniGrid-UnlockPickup-v0", *args, **kwargs)
    env.reset()

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
    # Dummy function returning fixed input
    make_task_fn = lambda env: tasks
    env = MinigridTaskEnvWrapper(env, make_task_fn)
    return env


def make_wrapped_pickup_unlock_env(*args, **kwargs):

    env = gym.make("MiniGrid-UnlockPickup-v0", *args, **kwargs)

    def make_tasks(env):
        tasks = [None] * 6
        for object in env.grid.grid:
            if object is None:
                continue
            elif object.type == "key":
                key_desc = ObjectDescription(object.type, object.color)
                tasks[0] = GoToObjectTask(key_desc)
                tasks[1] = PickUpObjectTask(key_desc)
            elif object.type == "door":
                door_desc = ObjectDescription(object.type, object.color)
                tasks[2] = GoToObjectTask(door_desc)
                tasks[3] = OpenDoorTask(door_desc)
            elif object.type == "box":
                box_desc = ObjectDescription(object.type, object.color)
                tasks[4] = GoToObjectTask(box_desc)
                tasks[5] = PickUpObjectTask(box_desc)
        return tasks

    env = MinigridTaskEnvWrapper(env, make_tasks)
    return env


if __name__ == "__main__":
    env = make_wrapped_pickup_unlock_env(render_mode="human")
    obs, _ = env.reset()

    print(obs["mission"])

    env.render()
    input("Press enter to continue...")
