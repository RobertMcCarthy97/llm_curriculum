import gymnasium as gym
from typing import Dict, List, Callable
from llm_curriculum.envs.minimal_minigrid.prompting.prompt import (
    parse_agent,
    parse_field_of_view,
)


class DecomposedRewardWrapper(gym.Wrapper):
    """
    Note: currently only works with FullyObsWrapper
    """

    def __init__(self, env, objectives: List[str], reward_functions: List[Callable]):
        """objectives"""
        super().__init__(env)
        self.objectives = objectives
        self.reward_functions = reward_functions
        self.current_objective_idx = 0

    def reset(self):
        obs, info = self.env.reset()
        self.current_objective_idx = 0
        obs["mission"] = self.objectives[self.current_objective_idx]
        return obs, info

    def get_current_objective(self):
        return self.objectives[self.current_objective_idx]

    def get_current_reward_function(self):
        return self.reward_functions[self.current_objective_idx]

    def all_subtasks_complete(self):
        return self.current_objective_idx == len(self.objectives) - 1

    def step(self, action):
        obs, orig_rew, term, trunc, info = self.env.step(action)
        if self.all_subtasks_complete():
            return obs, orig_rew, term, trunc, info

        # Overwrite the mission with the current objective
        obs["mission"] = self.get_current_objective()

        # Shape the reward with intermediate objectives
        function_obs = {
            "agent_info": parse_agent(self.env),
            "field_of_view": parse_field_of_view(obs["image"]),
        }
        objective_completion = self.get_current_reward_function()(function_obs)
        if objective_completion:
            self.current_objective_idx += 1
            sub_reward = 1
        else:
            sub_reward = 0

        return obs, sub_reward, term, trunc, info
