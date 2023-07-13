import gymnasium as gym
from typing import Dict, List, Callable
from llm_curriculum.envs.minimal_minigrid.prompting.prompt import (
    parse_agent,
    parse_field_of_view,
)
from llm_curriculum.envs.minimal_minigrid.envs.tasks import BaseTask
import collections
import numpy as np

from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX


class FullyObsInfoWrapper(gym.Wrapper):
    """Put the full observation in the info dict"""

    def __init__(self, env):
        super().__init__(env)

    @staticmethod
    def get_full_obs(env):
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )
        return {"image": full_grid}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["full_grid"] = self.get_full_obs(self.env)
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        info["full_grid"] = self.get_full_obs(self.env)
        return obs, rew, term, trunc, info


class DecomposedRewardWrapper(gym.Wrapper):
    """
    Note: currently only works with FullyObsInfoWrapper
    """

    def __init__(
        self,
        env,
        objectives: List[str],
        reward_functions: List[Callable],
        enable_mission: bool = True,
        enable_reward: bool = True,
    ):
        """
        :param: enable_mission: whether to overwrite default mission with subtasks
        :param: enable_reward: whether to shape default reward with subtasks
        """
        super().__init__(env)
        self.objectives = objectives
        self.reward_functions = reward_functions
        self.current_objective_idx = 0
        self.enable_mission = enable_mission
        self.enable_reward = enable_reward

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_objective_idx = 0

        if self.enable_mission:
            obs["mission"] = self.objectives[self.current_objective_idx]

        return obs, info

    def get_current_objective(self):
        return self.objectives[self.current_objective_idx]

    def get_current_reward_function(self):
        return self.reward_functions[self.current_objective_idx]

    def all_subtasks_complete(self):
        return self.current_objective_idx == len(self.objectives)

    @staticmethod
    def get_reward_function_obs(env, full_obs):
        """
        For now, we are assuming reward function has access to full information
        TODO: investigate reward function from partial observability in the future
        """

        def make_object():
            return {"position": (-1, -1)}

        field_of_view = collections.defaultdict(make_object)
        field_of_view.update(parse_field_of_view(full_obs["image"]))
        return {
            "agent_info": parse_agent(env.unwrapped),
            "field_of_view": field_of_view,
        }

    def step(self, action):
        obs, orig_rew, term, trunc, info = self.env.step(action)
        assert (
            "full_grid" in info
        ), "DecomposedRewardWrapper requires FullyObsInfoWrapper"
        full_obs = info["full_grid"]
        if self.all_subtasks_complete():
            return obs, orig_rew, term, trunc, info

        sub_reward = orig_rew
        if self.enable_reward:
            # Shape the reward with intermediate objectives
            function_obs = self.get_reward_function_obs(self.env, full_obs)
            try:
                objective_completion = self.get_current_reward_function()(function_obs)
                if objective_completion:
                    self.current_objective_idx += 1
                    sub_reward += 1
            except Exception as e:
                print(e)
                print("Failed to execute reward function. Skipping...")
                self.current_objective_idx += 1

        # Overwrite the mission with the current objective
        if self.enable_mission:
            if not self.all_subtasks_complete():
                obs["mission"] = self.get_current_objective()

        return obs, sub_reward, term, trunc, info


class OracleRewardWrapper(gym.Wrapper):
    """Wrap a Minigrid environment with a sequence of tasks

    Replace 'mission' with subtask
    Replace reward with subtask reward
    """

    def __init__(
        self,
        env: gym.Env,
        make_tasks_fn: Callable[[gym.Env], List[BaseTask]],
        enable_mission: bool = True,
        enable_reward: bool = True,
    ):
        super().__init__(env)
        self.make_tasks_fn = make_tasks_fn
        self.enable_mission = enable_mission
        self.enable_reward = enable_reward

    def get_current_task(self):
        return self.tasks[self.current_task_idx]

    def has_tasks_remaining(self):
        return self.current_task_idx < len(self.tasks)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.tasks = self.make_tasks_fn(self.env)
        self.current_task_idx = 0

        if self.enable_mission:
            info["overall_mission"] = obs["mission"]
            obs["mission"] = self.get_current_task().to_string()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # If no subtasks remain
        if not self.has_tasks_remaining():
            return obs, reward, terminated, truncated, info

        # else
        if self.enable_reward:
            # If subtask is completed
            task = self.get_current_task()
            task_success = task.check_success(self.env)
            if task_success:
                reward += 1
                self.current_task_idx += 1

        if self.enable_mission and self.has_tasks_remaining():
            info["overall_mission"] = obs["mission"]
            obs["mission"] = self.get_current_task().to_string()

        return obs, reward, terminated, truncated, info
