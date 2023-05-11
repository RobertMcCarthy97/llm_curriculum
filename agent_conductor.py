import numpy as np

from tasks import MoveCubeToTargetTask, PickUpCubeTask, GraspCubeTask
from sentence_transformers import SentenceTransformer

class StatsTracker():
    def __init__(self, task_names) -> None:
        self.task_names = task_names
        
        self.raw_stats = {}
        self.raw_stats['all'] = self.init_raw_stats(task_names)
        self.raw_stats['epoch'] = self.init_raw_stats(task_names)
    
    def init_raw_stats(self, task_names):
        raw_stats = {}
        for name  in task_names:
            raw_stats[name] = []
        return raw_stats
    
    def append_stat(self, task_name, stat):
        self.raw_stats['all'][task_name].append(stat)
        self.raw_stats['epoch'][task_name].append(stat)
        
    def reset_epoch_stats(self):
        self.raw_stats['epoch'] = self.init_raw_stats(self.task_names)
        
    def get_agg_stats(self, agg_func):
        agg_stats = {}
        for time_key in ['all', 'epoch']:
            agg_stats[time_key] = {}
            for task_key in self.raw_stats[time_key].keys():
                if len(self.raw_stats[time_key][task_key]) < 1:
                    agg_stat = None
                else:
                    agg_stat = agg_func(self.raw_stats[time_key][task_key])
                agg_stats[time_key][task_key] = agg_stat
        return agg_stats
    

class AgentConductor():
    def __init__(self, env, manual_decompose_p=None, dense_rew_lowest=False, single_task_names=None, high_level_task_names=None, contained_sequence=False):
        self.env = env
        self.manual_decompose_p = manual_decompose_p
        self.dense_rew_lowest = dense_rew_lowest
        
        self.single_task_names = single_task_names
        
        self.high_level_task_names = high_level_task_names
        if self.single_task_names is not None:
            assert self.manual_decompose_p == 1
            
        self.contained_sequence = contained_sequence
        if self.contained_sequence:
            assert self.single_task_names is not None and len(self.single_task_names) == 1
        
        # tasks
        self.high_level_task_list = self.init_possible_tasks(env)
        self.task_names = self.get_task_names()
        self.task_idx_dict, self.n_tasks = self.init_oracle_goals()
        
        # check single tasks are contained within high-level task tree
        if self.single_task_names is not None:
            assert len(self.high_level_task_list) == 1, "only deal with this for now..."
            all_task_names = self.get_task_names()
            assert all([single_task in all_task_names for single_task in self.single_task_names]), "Single task not contained within high-level task"
        
        # stats
        self.task_stats = {}
        self.task_stats['success'] = StatsTracker(self.task_names)
        self.task_stats['chosen_n'] = StatsTracker(self.task_names)
        self.task_stats['length'] = StatsTracker(self.task_names)
        
        # language embeddings
        # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        # Or use same as 'Guided pretraining' paper? https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2 
        self.sentence_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.init_task_embeddings()
        
        # init (to be overrided)
        self.active_task = self.high_level_task_list[0]
        self.active_task_steps_active = 0
    
    def init_possible_tasks(self, env):
        high_level_tasks = []
        if 'move_cube_to_target' in self.high_level_task_names:
            high_level_tasks += [MoveCubeToTargetTask(use_dense_reward_lowest_level=self.dense_rew_lowest)]
        if 'pick_up_cube' in self.high_level_task_names:
            high_level_tasks += [PickUpCubeTask(use_dense_reward_lowest_level=self.dense_rew_lowest)]
        if 'grasp_cube' in self.high_level_task_names:
            high_level_tasks += [GraspCubeTask(use_dense_reward_lowest_level=self.dense_rew_lowest)]
        assert len(high_level_tasks) >= 0
        return high_level_tasks
    
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
    
    def get_task_names(self):
        def recursively_get_names(task):
            task_names = [task.name]
            if len(task.subtask_sequence) > 0:
                for subtask in task.subtask_sequence:
                    subtask_names = recursively_get_names(subtask)
                    task_names = task_names + subtask_names
            return task_names
        
        task_names = []
        for task in self.high_level_task_list:
            task_names = task_names + recursively_get_names(task)
            
        return task_names
    
    # def get_single_task_names(self):
    #     return self.single_task_names

    def get_active_single_task_name(self):
        return self.active_single_task_name

    def set_single_task_names(self, single_task_names):
        self.single_task_names = single_task_names
    
    def init_oracle_goals(self):
        task_idx_dict = {}
        for i, name in enumerate(self.task_names):
            task_idx_dict[name] = i
        return task_idx_dict, len(self.task_names)
        
    def reset(self):
        # TODO: required for stats but not good...
        prev_active_task = self.active_task
        # reset tasks (i.e. set complete=False)
        self.reset_tasks()
        # if doing single task per episode (or sequenced), sample from self.single_task_names
        if self.single_task_names is not None:
            self.active_single_task_name = np.random.choice(self.single_task_names)
        else:
            self.active_single_task_name = None
        # choose random high-level task from list
        self.chosen_high_level_task = np.random.choice(self.high_level_task_list)
        # choose active task
        self.active_task = self.decompose_task(self.chosen_high_level_task)
        # stats
        self.record_chosen_task(self.active_task, prev_active_task=prev_active_task, reset=True)
        
        return self.active_task
    
    def reset_tasks(self):
        for high_level_task in self.high_level_task_list:
            high_level_task.reset()
    
    def step(self):
        prev_active_task = self.active_task
        self.active_task = self.step_task_recursive(prev_active_task)
        # stats
        self.record_chosen_task(self.active_task, prev_active_task=prev_active_task)
        return self.active_task
    
    def decompose_task(self, task):
        # Never decompose single task or contained sequence
        if task.name == self.active_single_task_name:
            return task
        if not task.complete:
            if len(task.subtask_sequence) >= 0:
                if self.decide_decompose(task):
                    for subtask in task.subtask_sequence:
                        if not subtask.complete:
                            return self.decompose_task(subtask)
        # Return current task (i.e. don't decompose) if (i) is complete, (ii) has no subtasks, or (iii) decided not to decompose
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
        if task.name == self.active_single_task_name:
            # contained sequence logic
            if self.contained_sequence:
                if task.complete and task.check_next_task_exists():
                    # jump to next task in sequence
                    self.active_single_task_name = task.next_task.name
                    return task.next_task
            # single task logic
            return task
        # normal logic
        if task.complete:
            if task.parent_task is None:
                assert task.check_next_task_exists() is False, "Can't have next task if no parent task."
                # If no parent, already at highest-level so stick with same
                return task
            else:
                # If completed task and parent exists - replan from start! (don't stay on same level)
                return self.decompose_task(self.chosen_high_level_task)
                # if task.check_next_task_exists() is False:
                #     # If completed sequence and parent exists - replan from start! (don't stay on same level)
                #     return self.decompose_task(self.chosen_high_level_task)
                # else:
                #     # If current complete and next exists -> go to next
                #     return self.step_task_recursive(task.get_next_task())
        else:
            # If task not complete, then keep trying!
            return task
    
    def get_oracle_action(self, state, task):
        direction_act, gripper_act = task.get_oracle_action(state)
        return FetchAction(self.env, direction_act, gripper_act).get_action()
    
    def record_chosen_task(self, chosen_task, prev_active_task=None, reset=False):
        # check if task changed
        if prev_active_task is not None:
            task_changed = (chosen_task.name != prev_active_task.name) # this isn't a good check...
        if task_changed or reset:
            # record change
            self.task_stats['chosen_n'].append_stat(chosen_task.name, 1)
            # record prev task active length
            self.task_stats['length'].append_stat(prev_active_task.name, self.active_task_steps_active)
            # reset counter
            self.active_task_steps_active = 0
        else:
            self.active_task_steps_active += 1
        
    def record_task_success_stat(self, task, is_success):
        self.task_stats['success'].append_stat(task.name, is_success)
    
    def get_task_success_rate(self, task):
        raise NotImplementedError()
    
    def get_stats(self):
        stats = {}
        stats['success'] = self.task_stats['success'].get_agg_stats(lambda x: np.mean(x))
        stats['chosen_n'] = self.task_stats['chosen_n'].get_agg_stats(lambda x: len(x))
        stats['length'] = self.task_stats['length'].get_agg_stats(lambda x: np.mean(x))
        return stats
    
    def reset_epoch_stats(self):
        for key in self.task_stats.keys():
            self.task_stats[key].reset_epoch_stats()
    
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