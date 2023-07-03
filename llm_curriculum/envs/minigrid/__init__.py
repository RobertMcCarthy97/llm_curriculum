from gymnasium import register

register(
    id="MiniGrid-UnlockPickupDecomposed-v0",
    entry_point="llm_curriculum.envs.minigrid.env_wrapper:make_wrapped_pickup_unlock_env",
)

register(
    id="MiniGrid-UnlockPickupDecomposedAutomated-v0",
    entry_point="llm_curriculum.envs.minigrid.env_wrapper:make_automated_env",
    args=("MiniGrid-UnlockPickup-v0",),
)

register(
    id="MiniGrid-BlockedUnlockPickupDecomposedAutomated-v1",
    entry_point="llm_curriculum.envs.minigrid.env_wrapper:make_automated_env",
    args=("MiniGrid-BlockedUnlockPickup-v0",),
)

register(
    id="MiniGrid-UnlockDecomposedAutomated-v1",
    entry_point="llm_curriculum.envs.minigrid.env_wrapper:make_automated_env",
    args=("MiniGrid-Unlock-v0",),
)
