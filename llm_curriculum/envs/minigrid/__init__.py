from gymnasium import register

register(
    id="MiniGrid-UnlockPickupDecomposed-v0",
    entry_point="llm_curriculum.envs.minigrid.env_wrapper:make_wrapped_pickup_unlock_env",
)

register(
    id="MiniGrid-UnlockDecomposedAutomated-v1",
    entry_point="llm_curriculum.envs.minigrid.env_wrapper:make_automated_unlock_env",
)

register(
    id="MiniGrid-UnlockPickupDecomposedAutomated-v0",
    entry_point="llm_curriculum.envs.minigrid.env_wrapper:make_automated_unlock_pickup_env",
)

register(
    id="MiniGrid-BlockedUnlockPickupDecomposedAutomated-v1",
    entry_point="llm_curriculum.envs.minigrid.env_wrapper:make_automated_blocked_unlock_pickup_env",
)
