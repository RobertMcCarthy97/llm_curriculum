import gym as gym_old


class NonGoalNonDictObsWrapper(gym_old.ObservationWrapper):
    """
    Recieve env with dict observations and only return state obs
    """

    def __init__(self, env):
        super().__init__(env)
        self._env = env

        self.observation_space = env.observation_space["observation"]

    def observation(self, observation):
        return observation["observation"]
