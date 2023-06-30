import gymnasium as gym

from typing import List
from llm_curriculum.envs.minigrid.tasks import BaseTask


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
