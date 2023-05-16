import gymnasium as gym
import gym as gym_old

import numpy as np
import time

from llm_curriculum_algo.agent_conductor import AgentConductor


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
    """
    Add the position of the red target to the observartion space
    """
    def __init__(self, env):
        super().__init__(env)
        self._env = env
        
        new_obs_size = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_size,))
        
    def observation(self, observation):
        obs = observation['observation']
        goal = observation['desired_goal']
        return np.concatenate([obs, goal], axis=0)


class NonGoalNonDictObsWrapper(gym_old.ObservationWrapper):
    """
    Recieve env with dict observations and only return state obs
    """
    def __init__(self, env):
        super().__init__(env)
        self._env = env
        
        self.observation_space = env.observation_space['observation']
        
    def observation(self, observation):
        return observation['observation']


class MTEnvWrapper(gym_old.Wrapper):
    def __init__(self, env, task_index: int):
        super().__init__(env)
        self._env = env
        self._task_index = task_index
        
        self.observation_space = gym_old.spaces.Dict({
            'env_obs': env.observation_space['observation'],
            'task_obs': gym_old.spaces.Box(low=0, high=np.inf, shape=(1,))
        })
        
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
        return {'env_obs': observation['observation'], 'task_obs': task_obs}


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
        self.reset_conductor = AgentConductor(env, manual_decompose_p=1, high_level_task_names=agent_conductor.high_level_task_names) # TODO: what if using different high-level task
        
        # begin counter
        self.active_task_steps = 0
        self.episode_n_steps = 0
        
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
        prev_task = self.agent_conductor.get_active_task()
        # single task scenario
        if len(self.agent_conductor.single_task_names) > 0:
            obs, info = self.reset_single_task(**kwargs) # TODO: need to stop task info tracking??
        else:
            obs, info = self.reset_normal(**kwargs)
        # stats
        self.episode_n_steps = 0
        active_task = self.agent_conductor.get_active_task()
        self.record_task_stats(prev_task, active_task, reset=True)
        assert active_task.name == info['active_task_name']
        # return
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
        info = self.set_info(active_task, active_task, state_obs)
        
        return obs, info
    
    def reset_single_task(self, **kwargs):
        '''
        Use agent oracle actions to step through subtask until reached the single task
        
        TODO: extend as follows:
        - take 'task_to_reset_to' as input
        - replace self.agent_conductor.active_single_task_name with 'task_to_reset_to'
        - 'task_to_reset_to' can just be high_level_task_name if no single tasks???
        '''
        for _ in range(50):
            obs, info = self.reset_normal(**kwargs)
            self.reset_conductor.set_single_task_names([self.agent_conductor.active_single_task_name])
            self.reset_conductor.reset()
            for _ in range(50):
                # check if reached desired task
                if self.reset_conductor.get_active_task().name == self.agent_conductor.get_active_single_task_name():
                    return obs, info
                reset_prev_active_task = self.reset_conductor.get_active_task()
                # action and step env
                action = self.reset_conductor.get_oracle_action(obs['observation'], reset_prev_active_task)
                obs, _, _, truncated, info = self.step(action, reset_step=True) # Don't record till reset
                _, reset_success = self.calc_reward(obs['observation'], reset_prev_active_task)
                # step reset_conductor
                reset_active_task = self.reset_conductor.step()
                reset_goal_changed = (reset_prev_active_task.name != reset_active_task.name)
                if (reset_success and reset_goal_changed) or truncated:
                    self.reset_conductor.record_task_success_stat(reset_prev_active_task, reset_success)
        assert False, "Failed to reach single task"
        
    
    def step(self, action, reset_step=False):
        # TODO: fix reward? currently r = f(s_t+1, g_t)
        # prev active task
        prev_active_task = self.agent_conductor.get_active_task()
        
        # step env
        state_obs, _, _, _, _ = self._env.step(action)
        self.active_state_obs = state_obs
        
        # calc reward
        reward, success = self.calc_reward(state_obs, prev_active_task) # reward based on g_t, obs_t+1
        
        # replan
        active_task = self.agent_conductor.step()
        
        # obs
        obs = self.set_obs(state_obs, active_task)
        
        # info
        info = self.set_info(prev_active_task, active_task, state_obs)
        info['is_success'], info['success'] = success, success
        info['goal_changed'] = (prev_active_task.name != active_task.name)
        assert active_task.name == info['active_task_name']
        
        if not reset_step:
            self.episode_n_steps += 1
        # terminated
        terminated = False       
        # truncated
        truncated = (self.episode_n_steps >= self.max_ep_len)
        
        # stats
        if not reset_step:
            # record stats
            self.record_task_stats(prev_active_task, active_task)
            if (success and info['goal_changed']) or truncated:
                self.agent_conductor.record_task_success_stat(prev_active_task, success)        
        
        return obs, reward, terminated, truncated, info

    def record_task_stats(self, prev_task, active_task, reset=False):
        # check if task changed
        task_changed = (prev_task.name != active_task.name)
        if (task_changed or reset) and (self.active_task_steps > 0):
            # record previous task count and n_steps
            self.agent_conductor.record_task_chosen_stat(prev_task, self.active_task_steps)
            # reset steps
            self.active_task_steps = 0
        else:
            # increment steps
            self.active_task_steps += 1
    
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
    
    def set_info(self, prev_task, stepped_task, current_state):
        info = {}
        # prev task details
        info['prev_task_level'] = prev_task.level
        info['prev_task_name'] = prev_task.name
        # active task details
        info['active_task_level'] = stepped_task.level
        info['active_task_name'] = stepped_task.name
        # info['active_task_index'] = self.agent_conductor.get_task_index(stepped_task)
        # record parent tasks details
        iter_task = stepped_task
        for i in range(stepped_task.level, -1, -1):
            info[f'task_level_{i}'] = iter_task.name
            iter_task = iter_task.parent_task
        # record parent task reward and goal
        if prev_task.parent_task is not None: # TODO: delete this?
            info['obs_parent_goal_reward'] = prev_task.parent_task.check_success_reward(current_state)[1] # parent reward depends on parent_g_t, obs_t+1 - TODO: this naming is very confusding!!
            info['obs_parent_goal'] = self.get_task_goal(prev_task.parent_task)
        for parent_name in prev_task.relations['parents'].keys():
            info[f'obs_{parent_name}_reward'] = self.agent_conductor.get_task_from_name(parent_name).check_success_reward(current_state)[1]
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

def make_env(manual_decompose_p=1, dense_rew_lowest=False, dense_rew_tasks=[], use_language_goals=False, render_mode=None, max_ep_len=50, single_task_names=[], high_level_task_names=None, contained_sequence=False, state_obs_only=False, mtenv_wrapper=False, mtenv_task_idx=None, curriculum_manager_cls=None):
    
    env = gym.make("FetchPickAndPlace-v2", render_mode=render_mode)
    env = AddTargetToObsWrapper(env)
    
    agent_conductor = AgentConductor(env, manual_decompose_p=manual_decompose_p, dense_rew_lowest=dense_rew_lowest, dense_rew_tasks=dense_rew_tasks, single_task_names=single_task_names, high_level_task_names=high_level_task_names, contained_sequence=contained_sequence, use_language_goals=use_language_goals)
    env = CurriculumEnvWrapper(env, agent_conductor, use_language_goals=use_language_goals, max_ep_len=max_ep_len)
    if curriculum_manager_cls is not None:
        curriculum_manager = curriculum_manager_cls(tasks_list=agent_conductor.get_possible_task_names(), agent_conductor=agent_conductor)
        agent_conductor.set_curriculum_manager(curriculum_manager) # TODO: not good stuff

    env = OldGymAPIWrapper(env)
    if state_obs_only:
        env = NonGoalNonDictObsWrapper(env)
    if mtenv_wrapper:
        env = MTEnvWrapper(env, mtenv_task_idx)
    return env

def make_env_baseline(name="FetchPickAndPlace-v2", render_mode=None, max_ep_len=50):
    env = gym.make(name, render_mode=render_mode)
    env = OldGymAPIWrapper(env, max_ep_len)
    return env

if __name__ == "__main__":
    from llm_curriculum_algo.curriculum_manager import DummySeperateEpisodesCM, SeperateEpisodesCM
    
    env = make_env(
        manual_decompose_p=0.5,
        dense_rew_lowest=False,
        dense_rew_tasks=["move_gripper_to_cube", "move_cube_towards_target_grasp"],
        use_language_goals=False,
        render_mode="human",
        single_task_names=[],
        high_level_task_names=["move_cube_to_target"],
        contained_sequence=False,
        curriculum_manager_cls=SeperateEpisodesCM
    )
    
    # env = make_env(
    #     manual_decompose_p=1,
    #     dense_rew_lowest=False,
    #     dense_rew_tasks=[],
    #     use_language_goals=False,
    #     render_mode="human",
    #     single_task_names=["lift_cube"],
    #     high_level_task_names=["move_cube_to_target"],
    #     contained_sequence=False,
    #     curriculum_manager_cls=SeperateEpisodesCM
    # )

    for _ in range(5):
        
        # obs, info = env.reset()
        obs = env.reset()
        print("env reset")

        for _ in range(35):
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
            # print(f"info: {info}")
            # print("Parent goal: ", info.get('obs_parent_goal', None))
            # print("Parent goal reward: ", info.get('obs_parent_goal_reward', None))
            print()
            
            # # env.render()
            # time.sleep(0.1)
            
        stats = env.agent_conductor.get_stats()
        print(stats)
        env.agent_conductor.reset_epoch_stats()
