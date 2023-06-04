import numpy as np
import gymnasium as gym
import gym as gym_old


class MTEnvWrapper(gym_old.Wrapper):
    def __init__(self, env, task_index: int):
        super().__init__(env)
        self._env = env
        self._task_index = task_index

        self.observation_space = gym_old.spaces.Dict(
            {
                "env_obs": env.observation_space["observation"],
                "task_obs": gym_old.spaces.Box(low=0, high=np.inf, shape=(1,)),
            }
        )

    def reset(self):
        # TODO: make sure correct task idx is set here...
        obs = self._env.reset()
        return self.mod_observation(obs)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return self.mod_observation(obs), reward, done, info

    def mod_observation(self, observation):
        task_obs = np.array([self._task_index])
        # assert task_idx == info['active_task_idx'] # TODO
        return {"env_obs": observation["observation"], "task_obs": task_obs}
