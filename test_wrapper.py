import gymnasium as gym

from sentence_transformers import SentenceTransformer

import numpy as np
import time

from tasks import MoveCubeToTargetTask

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
    def __init__(self, env, agent_conductor, use_language_goals=False, oracle_n_action_seed=0):
        super().__init__(env)
        self._env = env
        self.use_language_goals = use_language_goals
        self.oracle_n_action_seed = oracle_n_action_seed
        
        self.step_count = 0
        self.agent_conductor = agent_conductor
        self.init_obs_space()
        
    def init_obs_space(self):
        # goal observation
        if self.use_language_goals:
            goal_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(384,), dtype=np.float32)
        else:
            n_tasks = self.agent_conductor.n_tasks
            goal_space = gym.spaces.Box(low=0.0, high=1.0, shape=(n_tasks,), dtype=np.float32)
        # new dict observation space
        self.observation_space = gym.spaces.Dict({
                'observation': self._env.observation_space,
                'desired_goal': goal_space,
            })

    def reset(self, **kwargs):
        state_obs, info = self._env.reset(**kwargs)
        # task
        active_task = self.agent_conductor.reset_task()
        # obs
        obs = self.set_obs(state_obs, active_task)
        # info
        info = self.set_info(active_task)
        
        return obs, info
    
    def step(self, action):
        # prev active task
        prev_active_task = self.agent_conductor.get_active_task()
        # oracle_action
        if self.step_count < self.oracle_n_action_seed:
            action = self.agent_conductor.get_oracle_action(prev_active_task)
        # step env
        state_obs, _, _, truncated, _ = self._env.step(action)
        
        # calc reward
        reward, success = self.calc_reward(state_obs, prev_active_task)
        # replan
        active_task = self.agent_conductor.step_task()
        
        # obs
        obs = self.set_obs(state_obs, active_task)
        # info
        info = self.set_info(active_task, obs['observation'])
        info['is_success'] = success
        info['goal_changed'] = (prev_active_task.name != active_task.name)
        # terminated
        terminated = False # TODO: implement termination condition
        
        # stats
        if (success and info['goal_changed']) or truncated:
            self.agent_conductor.record_task_success_stat(prev_active_task, success)
        self.step_count += 1
        
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
        info['active_task_level'] = active_task.level
        info['active_task_name'] = active_task.name
        # record parent tasks
        iter_task = active_task
        for i in range(active_task.level, -1, -1):
            info[f'task_level_{i}'] = iter_task.name
            iter_task = iter_task.parent_task
        # record parent task reward and goal
        _, parent_reward = active_task.parent_task.check_success_reward(current_state)
        parent_goal = self.get_task_goal(active_task.parent_task)
        info['parent_task_reward'] = parent_reward
        info['parent_goal'] = parent_goal
        # return
        return {'task_info': info}
    
    def calc_reward(self, state, active_task):
        success, reward = active_task.active_task_check_and_set_success(state)
        return reward, success
        
    def plot_state(self):
        # plot the state of the environment, for debugging purposes
        pass
    
    def get_oracle_action(self, state):
        return self.agent_conductor.get_oracle_action(state, self.agent_conductor.active_task)
        

class AgentConductor():
    def __init__(self, env, manual_decompose_p=None, dense_rew_lowest=False):
        self.env = env
        self.manual_decompose_p = manual_decompose_p
        self.dense_rew_lowest = dense_rew_lowest
        
        self.high_level_task_list = self.init_possible_tasks(env)
        self.task_success_stats = self.init_task_success_stats()
        self.task_idx_dict, self.n_tasks = self.init_oracle_goals()
        
        # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        # Or use same as 'Guided pretraining' paper? https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2 
        self.sentence_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.init_task_embeddings()
    
    def init_possible_tasks(self, env):
        return [MoveCubeToTargetTask(use_dense_reward_lowest_level=self.dense_rew_lowest)]
    
    def init_task_embeddings(self):
        def recursively_get_embeddings(task):
            task_name = task.get_str_description()
            task_embedding = self.sentence_embedder.encode([task_name])[0]
            task_embeddings_dict = {task_name: task_embedding}
            if len(task.subtask_sequence) > 0:
                for subtask in task.subtask_sequence:
                    subtask_embeddings_dict = recursively_get_embeddings(subtask)
                    task_embeddings_dict.update(subtask_embeddings_dict)
            return task_embeddings_dict
            
        task_embeddings_dict = {}
        for task in self.high_level_task_list:
            task_embeddings_dict.update(recursively_get_embeddings(task))
            
        return task_embeddings_dict
    
    def init_task_success_stats(self):
        def recursively_init_success_stats(task):
            task_name = task.name
            task_success_stats = {task_name: [0]}
            if len(task.subtask_sequence) > 0:
                for subtask in task.subtask_sequence:
                    subtask_success_stats = recursively_init_success_stats(subtask)
                    task_success_stats.update(subtask_success_stats)
            return task_success_stats
        
        task_success_stats = {}
        for task in self.high_level_task_list:
            task_success_stats.update(recursively_init_success_stats(task))
            
        return task_success_stats
    
    def init_oracle_goals(self):
        task_names = list(self.task_success_stats.keys())
        task_idx_dict = {}
        for i, name in enumerate(task_names):
            task_idx_dict[name] = i
        return task_idx_dict, len(task_names)
        
    def reset_task(self):
        # choose random high-level task from list
        self.chosen_high_level_task = np.random.choice(self.high_level_task_list)
        # choose active task
        self.active_task = self.decompose_task(self.chosen_high_level_task)
        
        return self.active_task
    
    def step_task(self):
        self.active_task = self.step_task_recursive(self.active_task)
        return self.active_task
    
    def decompose_task(self, task):
        if not task.complete:
            if len(task.subtask_sequence) >= 0:
                if self.decide_decompose(task):
                    for subtask in task.subtask_sequence:
                        if not subtask.complete:
                            return self.decompose_task(subtask)
        # Return current task if (i) is complete, (ii) has no subtasks, or (iii) decided not to decompose
        return task
    
    def decide_decompose(self, task):
        if self.manual_decompose_p is None:
            task_success_rate = self.get_task_success_rate(task)
            decompose_p = np.clip(0.9 - task_success_rate, 0.05, 1)
            do_decompose = np.random.choice([True, False], p=[decompose_p, 1-decompose_p])
        else:
            p = [self.manual_decompose_p, 1-self.manual_decompose_p]
            do_decompose = np.random.choice([True, False], p=p)
        return do_decompose
    
    def step_task_recursive(self, task):
        if task.complete:
            if task.parent_task is None:
                assert task.next_task is None, "Can't have next task if no parent task."
                # If no parent, already at highest-level so stick with same
                return task
            else:
                if task.next_task is None:
                    # If completed sequence and parent exists - replan from start! (don't stay on same level)
                    return self.decompose_task(self.chosen_high_level_task)
                else:
                    # If current complete and next exists -> go to next
                    return self.step_task_recursive(task.next_task)
        else:
            # If task not complete, then keep trying!
            return task
    
    def get_oracle_action(self, state, task):
        direction_act, gripper_act = task.get_oracle_action(state)
        return FetchAction(self.env, direction_act, gripper_act).get_action()
    
    def record_task_success_stat(self, task, is_success):
        self.task_success_stats[task.name].append(is_success)
    
    def get_task_success_rate(self, task):
        task_name = task.name
        task_success_stats = self.task_success_stats[task_name]
        return np.mean(task_success_stats)
    
    def get_task_embedding(self, task):
        task_name = task.get_str_description()
        return self.task_embeddings_dict[task_name]
    
    def get_task_oracle_goal(self, task):
        task_idx = self.task_idx_dict[task.name]
        one_hot = np.zeros(self.n_tasks)
        one_hot[task_idx] = 1
        return one_hot
    
    def get_active_task(self):
        return self.active_task


class FetchAction():
    def __init__(self, env, direction, gripper_open):
        self.env = env
        self.action_direction = np.clip(direction * 10, -1.0, 1.0)
        if gripper_open:
            self.action_gripper = np.array([1])
        else:
            self.action_gripper = np.array([-1])
            
        self.env_action = np.concatenate([self.action_direction, self.action_gripper]).astype('float32')
        assert self.env_action in self.env.action_space
        
    def get_action(self):
        return self.env_action

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
    

if __name__ == "__main__":
    
    env = gym.make("FetchPickAndPlace-v2", render_mode="human")
    env = AddTargetToObsWrapper(env)

    agent_conductor = AgentConductor(env, manual_decompose_p=1, dense_rew_lowest=True)
    env = CurriculumEnvWrapper(env, agent_conductor, use_language_goals=False)

    # temp
    # env.agent_conductor.task_success_stats['place_cube_at_target'] = [1] * 100

    obs, info = env.reset()

    for _ in range(50):
        ## Actions
        # action = env.action_space.sample()
        # action = get_user_action()
        action = env.get_oracle_action(obs['observation'])
        # input()
        # print(action)
        
        # step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # prints
        print(f"Reward: {reward}")
        active_task = info['task_info']['active_task']
        print(f"Active Task: {active_task}")
        print(f"Goal: {obs['desired_goal']}")
        print(f"Obs: {obs['observation'].shape}")
        print()
        
        # env.render()
        time.sleep(0.1)
