import gymnasium as gym
import gym as gym_old


class OldGymAPIWrapper(gym.Wrapper):
    """
    - Converts env ouputs to old gym API format
    - Assumes no terminations and enforces a max_ep_len truncation
    """

    def __init__(self, env):
        super().__init__(env)

        # action space
        assert isinstance(self.action_space, gym.spaces.Box)
        self.action_space = gym_old.spaces.Box(
            low=self.action_space.low,
            high=self.action_space.high,
            shape=self.action_space.shape,
        )

        ## Setup spaces
        # Box obs
        if isinstance(self.observation_space, gym.spaces.Box):
            self.observation_space = gym_old.spaces.Box(
                low=self.observation_space.low,
                high=self.observation_space.high,
                shape=self.observation_space.shape,
            )
        # Dict obs
        elif isinstance(self.observation_space, gym.spaces.Dict):
            new_space_dict = {}
            for key, space in self.observation_space.spaces.items():
                assert isinstance(self.observation_space[key], gym.spaces.Box)
                new_space_dict[key] = gym_old.spaces.Box(
                    low=space.low, high=space.high, shape=space.shape
                )

            self.observation_space = gym_old.spaces.Dict(new_space_dict)

    def reset(self):
        obs, info = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, _, truncated, info = self.env.step(action)
        # handle dones
        if truncated:
            done = True
            info["TimeLimit.truncated"] = True
        else:
            done = False
            info["TimeLimit.truncated"] = False
        # return expected items
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        return self.env.render()
