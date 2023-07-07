import minigrid
import gymnasium as gym

from llm_curriculum.envs.minigrid.env_wrapper import env_to_str

if __name__ == "__main__":

    env = gym.make("MiniGrid-UnlockPickup-v0", render_mode="human")
    obs, _ = env.reset()
    env.render()

    env_str = env_to_str(env)
    print(env_str)

    input()
