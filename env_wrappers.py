import gymnasium as gym
import gym as gym_old

import numpy as np
import time

from agent_conductor import AgentConductor


class OldGymAPIWrapper(gym.Wrapper):
    '''
    - Converts env ouputs to old gym API format
    - Assumes no terminations and enforces a max_ep_len truncation 
    '''
    def __init__(self, env):
        super().__init__(env)
        
        # action space
        assert isinstance(self.action_space, gym.spaces.Box)
        self.action_space = gym_old.spaces.Box(low=self.action_space.low, high=self.action_space.high, shape=self.action_space.shape)
        
        ## Setup spaces
        # Box obs
        if isinstance(self.observation_space, gym.spaces.Box):
            self.observation_space = gym_old.spaces.Box(low=self.observation_space.low, high=self.observation_space.high, shape=self.observation_space.shape)
        # Dict obs
        elif isinstance(self.observation_space, gym.spaces.Dict):
            new_space_dict = {}
            for key, space in self.observation_space.spaces.items():
                assert isinstance(self.observation_space[key], gym.spaces.Box)
                new_space_dict[key] = gym_old.spaces.Box(low=space.low, high=space.high, shape=space.shape)
            
            self.observation_space = gym_old.spaces.Dict(new_space_dict)
                
    def reset(self):
        obs, info = self.env.reset()
        self.ep_steps = 0
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
    
    def render(self, mode='rgb_array'):
        return self.env.render()


class AddTargetToObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._env = env
        
        new_obs_size = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_size,))
        
    def observation(self, observation):
        obs = observation['observation']
        goal = observation['desired_goal']
        return np.concatenate([obs, goal], axis=0)
    

class CurriculumEnvWrapper(gym.Wrapper):
    '''
    - Adds the target pos to observation
    - Goals are specified in language, and decomposed by agent_conductor
    '''
    def __init__(self, env, agent_conductor, use_language_goals=False, max_ep_len=50):
        super().__init__(env)
        self._env = env
        self.use_language_goals = use_language_goals
        self.agent_conductor = agent_conductor
        self.max_ep_len = max_ep_len
        
        self.init_obs_space()
        self.reset_conductor = AgentConductor(env, manual_decompose_p=1, high_level_task_names='move_cube_to_target')
        
    def init_obs_space(self):
        # goal observation
        if self.use_language_goals:
            goal_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(384,), dtype=np.float32)
        else:
            # assert False, "Not working for direction tasks"
            n_tasks = self.agent_conductor.n_tasks
            goal_space = gym.spaces.Box(low=0.0, high=1.0, shape=(n_tasks,), dtype=np.float32)
        # new dict observation space
        self.observation_space = gym.spaces.Dict({
                'observation': self._env.observation_space,
                'desired_goal': goal_space,
            })
    
    def reset(self, **kwargs):
        # single task scenario
        if self.agent_conductor.single_task_names is not None:
            obs, info =  self.reset_single_task(**kwargs) # TODO: need to stop task info tracking??
        else:
            obs, info =  self.reset_normal(**kwargs)
        self.episode_n_steps = 0
        return obs, info
    
    def reset_normal(self, **kwargs):
        # reset env
        state_obs, _ = self._env.reset(**kwargs)
        self.active_state_obs = state_obs
        # get task
        active_task = self.agent_conductor.reset()
        # obs
        obs = self.set_obs(state_obs, active_task)
        # info
        info = self.set_info(active_task, state_obs)
        
        return obs, info
    
    def reset_single_task(self, **kwargs):
        '''
        Use agent oracle actions to step through subtask until reached the single task
        '''
        for _ in range(50):
            obs, info = self.reset_normal(**kwargs)
            self.reset_conductor.reset()
            self.reset_conductor.active_single_task_name = self.agent_conductor.active_single_task_name
            assert self.reset_conductor.active_single_task_name != "move_cube_to_target", "resets broken for highest-level single task"
            for _ in range(50):
                reset_prev_active_task = self.reset_conductor.get_active_task()
                # print(f"\nreset active: {reset_prev_active_task.name}")
                # print(f"desired: {self.agent_conductor.active_single_task_name}")
                # input()
                # action and step env
                action = self.reset_conductor.get_oracle_action(obs['observation'], reset_prev_active_task)
                obs, _, _, truncated, info = self.step(action, reset_step=True)
                _, reset_success = self.calc_reward(obs['observation'], reset_prev_active_task)
                # step reset_conductor
                reset_active_task = self.reset_conductor.step()
                reset_goal_changed = (reset_prev_active_task.name != reset_active_task.name)
                if (reset_success and reset_goal_changed) or truncated:
                    self.reset_conductor.record_task_success_stat(reset_prev_active_task, reset_success)
                # check if reached desired task
                if self.reset_conductor.get_active_task().name == self.agent_conductor.active_single_task_name:
                    return obs, info
        assert False, "Failed to reach single task"
        
    
    def step(self, action, reset_step=False):
        # TODO: fix reward? currently r = f(s_t+1, g_t)
        # prev active task
        prev_active_task = self.agent_conductor.get_active_task()
        # step env
        state_obs, _, _, truncated, _ = self._env.step(action)
        self.active_state_obs = state_obs
        # calc reward
        reward, success = self.calc_reward(state_obs, prev_active_task)
        # replan
        active_task = self.agent_conductor.step()
        
        # obs
        obs = self.set_obs(state_obs, active_task)
        # info
        info = self.set_info(active_task, obs['observation'])
        info['is_success'] = success
        info['goal_changed'] = (prev_active_task.name != active_task.name)
        
        if not reset_step:
            self.episode_n_steps += 1
            # truncated
            truncated = (self.episode_n_steps >= self.max_ep_len)
        else:
            truncated = False
        # terminated
        terminated = False # TODO: implement termination condition
        
        # stats
        if (success and info['goal_changed']) or truncated:
            self.agent_conductor.record_task_success_stat(prev_active_task, success)
        
        return obs, reward, terminated, truncated, info

    def set_obs(self, state_obs, active_task):
        obs = {}
        obs['observation'] = state_obs
        # update desired goal in obs
        obs['desired_goal'] = self.get_task_goal(active_task)
        # TODO: implement achieved goal...
        return obs
    
    def get_task_goal(self, task):
        if self.use_language_goals:
            return self.agent_conductor.get_task_embedding(task)
        else:
            return self.agent_conductor.get_task_oracle_goal(task)
    
    def set_info(self, active_task, current_state):
        info = {}
        # active task details
        info['active_task_level'] = active_task.level
        info['active_task_name'] = active_task.name
        # record parent tasks details
        iter_task = active_task
        for i in range(active_task.level, -1, -1):
            info[f'task_level_{i}'] = iter_task.name
            iter_task = iter_task.parent_task
        # record parent task reward and goal
        if active_task.parent_task is not None:
            _, info['parent_task_reward'] = active_task.parent_task.check_success_reward(current_state)
            info['parent_goal'] = self.get_task_goal(active_task.parent_task)
        # record overall success
        info['overall_task_success'], _ = self.agent_conductor.chosen_high_level_task.check_success_reward(current_state)
        # return
        return info
    
    def calc_reward(self, state, active_task):
        success, reward = active_task.active_task_check_and_set_success(state)
        return reward, success
        
    def plot_state(self):
        # plot the state of the environment, for debugging purposes
        pass
    
    def get_oracle_action(self, state):
        return self.agent_conductor.get_oracle_action(state, self.agent_conductor.active_task)


def get_user_action():
    prompt = "Enter action: \n"
    prompt +=  "[F]orward, [B]ackward, [L]eft, [R]ight, [U]p, [D]own, [O]pen, [C]lose: "
    user_action = input(prompt)
    if user_action == 'F':
        action = np.array([1, 0, 0, 0])
    elif user_action == 'B':
        action = np.array([-1, 0, 0, 0])
    elif user_action == 'L':
        action = np.array([0, 1, 0, 0])
    elif user_action == 'R':
        action = np.array([0, -1, 0, 0])
    elif user_action == 'U':
        action = np.array([0, 0, 1, 0])
    elif user_action == 'D':
        action = np.array([0, 0, -1, 0])
    elif user_action == 'O':
        action = np.array([0, 0, 0, 1])
    elif user_action == 'C':
        action = np.array([0, 0, 0, -1])
    else:
        action = np.array([0, 0, 0, 0])
    return action * 0.5

def make_env(manual_decompose_p=1, dense_rew_lowest=True, use_language_goals=False, render_mode=None, max_ep_len=50, single_task_names=None, high_level_task_names=None, contained_sequence=False):
    
    env = gym.make("FetchPickAndPlace-v2", render_mode=render_mode)
    env = AddTargetToObsWrapper(env)
    agent_conductor = AgentConductor(env, manual_decompose_p=manual_decompose_p, dense_rew_lowest=dense_rew_lowest, single_task_names=single_task_names, high_level_task_names=high_level_task_names, contained_sequence=contained_sequence)
    env = CurriculumEnvWrapper(env, agent_conductor, use_language_goals=use_language_goals, max_ep_len=max_ep_len)
    env = OldGymAPIWrapper(env)
    return env

def make_env_baseline(name="FetchPickAndPlace-v2", render_mode=None, max_ep_len=50):
    env = gym.make(name, render_mode=render_mode)
    env = OldGymAPIWrapper(env, max_ep_len)
    return env

if __name__ == "__main__":
    
    env = make_env(
        manual_decompose_p=1,
        dense_rew_lowest=False,
        use_language_goals=False,
        render_mode="human",
        single_task_names=["move_gripper_to_cube"],
        high_level_task_names=["move_cube_to_target"],
        contained_sequence=False,
        max_ep_len=25
    )

    for _ in range(5):
        
        # obs, info = env.reset()
        obs = env.reset()

        for _ in range(25):
            ## Actions
            # action = env.action_space.sample()
            # action = get_user_action()
            action = env.get_oracle_action(obs['observation'])
            # print(action)
            input()
            
            # step
            # obs, reward, terminated, truncated, info = env.step(action)
            obs, reward, done, info = env.step(action)
            
            # prints
            active_task = info['active_task_name']
            print(f"Active Task: {active_task}")
            print(f"Goal: {obs['desired_goal']}")
            # print(f"Obs: {obs['observation'].shape}")
            # print(f"step count: {env.ep_steps}")
            print(f"success: {info['is_success']}")
            print(f"Reward: {reward}")
            print("done: ", done)
            print()
            
            # # env.render()
            # time.sleep(0.1)
            
        stats = env.agent_conductor.get_stats()
        print(stats)
        env.agent_conductor.reset_epoch_stats()
