from gymnasium import register

register(
    id="MiniGrid-UnlockPickupDecomposed-v0",
    entry_point="llm_curriculum.envs.minigrid.env_wrapper:make_wrapped_pickup_unlock_env",
)
