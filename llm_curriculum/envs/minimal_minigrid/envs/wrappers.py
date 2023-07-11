import gymnasium as gym
from typing import Dict, List, Callable
from llm_curriculum.envs.minimal_minigrid.prompting.prompt import (
    parse_agent,
    parse_field_of_view,
)
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

    def __init__(self, env, objectives: List[str], reward_functions: List[Callable]):
        """objectives"""
        super().__init__(env)
        self.objectives = objectives
        self.reward_functions = reward_functions
        self.current_objective_idx = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_objective_idx = 0
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
            "agent_info": parse_agent(env),
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

        # Shape the reward with intermediate objectives
        function_obs = self.get_reward_function_obs(self.env, full_obs)
        objective_completion = self.get_current_reward_function()(function_obs)
        if objective_completion:
            self.current_objective_idx += 1
            # Add back orig reward
            # Ensure we don't diminish original reward signal
            sub_reward = orig_rew + 1
        else:
            sub_reward = orig_rew

        # Overwrite the mission with the current objective
        if not self.all_subtasks_complete():
            obs["mission"] = self.get_current_objective()

        return obs, sub_reward, term, trunc, info
