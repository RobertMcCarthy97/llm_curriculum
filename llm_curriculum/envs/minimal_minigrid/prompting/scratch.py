import minigrid
import llm_curriculum.envs.minimal_minigrid.envs
import gymnasium as gym

from llm_curriculum.envs.minimal_minigrid.description import describe_env

if __name__ == "__main__":
    env = gym.make("MiniGrid-UnlockPickup-v0")
    obs, _ = env.reset()
    env_str = describe_env(env.unwrapped)
    print(env_str)
