import gymnasium as gym
import numpy as np

from typing import List, Callable

from minigrid.wrappers import ReseedWrapper
from llm_curriculum.envs.minigrid.tasks import BaseTask
from llm_curriculum.envs.minigrid.grid_utils import get_object_pos, ObjectDescription
from llm_curriculum.envs.minigrid.tasks import (
    GoToObjectTask,
    PickUpObjectTask,
    OpenDoorTask,
)


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

    def reset(self):
        obs, info = self.env.reset()
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
