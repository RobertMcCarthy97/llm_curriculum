import gymnasium as gym
from gymnasium import register

register(
    "MiniGrid-IsNextTo-6x6-N2-v0",
    entry_point="llm_curriculum.envs.minimal_minigrid.envs.is_next_to:IsNextToEnv",
)
