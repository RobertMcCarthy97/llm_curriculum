import gymnasium as gym

from typing import List
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

    def __init__(self, env, tasks: List[BaseTask]):
        super().__init__(env)
        self.tasks = tasks
        self.current_task_idx = 0

    def get_current_task(self):
        return self.tasks[self.current_task_idx]

    def step(self, action):
        base_obs, _, _, truncated, info = self.env.step(action)

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

        info["overall_mission"] = base_obs["mission"]
        base_obs["mission"] = self.get_current_task().to_string()

        return base_obs, reward, terminated, truncated, info


def make_wrapped_pickup_unlock_env(*args, **kwargs):

    env = gym.make("MiniGrid-UnlockPickup-v0", *args, **kwargs)
    env.reset()
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

    return MinigridTaskEnvWrapper(env, tasks)
