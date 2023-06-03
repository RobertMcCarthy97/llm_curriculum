import numpy as np
import gymnasium as gym 

if __name__ == '__main__':
    env = gym.make('FetchPickAndPlace-v2', render_mode = 'human')
    env.reset()
    for _ in range(100):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        print(obs.keys())
        print(obs['observation'].shape, obs['desired_goal'].shape, obs['achieved_goal'].shape)
        env.render()
        if term or trunc: break
    env.close()