import random
import numpy as np

from agent_conductor import StatsTracker

class CurriculumManager():
    def __init__(self, tasks_list, agent_conductor):
        self.tasks_list = tasks_list
        self.agent_conductor = agent_conductor
        
        self.last_returned_i = -1
        self.stat_tracker = StatsTracker(tasks_list)
        
    def __iter__(self):
        return self

    def __next__(self):
        return self.next_task()
        
    def next_task(self):
        raise NotImplementedError
    
    def get_agg_stats(self):
        return self.stat_tracker.get_agg_stats(lambda x: np.mean(x))
    
    def reset_epoch_stats(self):
        self.stat_tracker.reset_epoch_stats()


class DummySeperateEpisodesCM(CurriculumManager):
    def __init__(self, tasks_list, agent_conductor):
        super().__init__(tasks_list, agent_conductor)
        
    def next_task(self):
        i = self.last_returned_i + 1
        i = i % len(self.tasks_list)
        chosen_task = self.tasks_list[i]
        self.last_returned_i = i
        self.stat_tracker.append_stat(chosen_task, 1)
        return chosen_task


class SeperateEpisodesCM(CurriculumManager):
    def __init__(self, tasks_list, agent_conductor):
        super().__init__(tasks_list, agent_conductor)
        
        self.child_parent_info = self.init_child_parent_info()
        
    def init_child_parent_info(self):
        tasks_info = {}
        for task_name in self.tasks_list:
            task_info = {'children': [], 'parent': None}
            
            task = self.agent_conductor.get_task_from_name(task_name)
            
            if len(task.subtask_sequence) > 0:
                for child in task.subtask_sequence:
                    if child.name in self.tasks_list:
                        task_info['children'].append(child.name)
            
            if task.parent_task is not None:
                if task.parent_task.name in self.tasks_list:
                    task_info['parent'] = task.parent_task.name
                    
            tasks_info[task_name] = task_info
        
        return tasks_info

    def calculate_probability(self, task_name):
        p_list = []
        
        p_self = self.get_p_task(task_name, positive_relationship=False)
        p_list.append(p_self)
        
        if self.child_parent_info[task_name]['parent'] is not None:
            p_parent = self.get_p_task(self.child_parent_info[task_name]['parent'], positive_relationship=False)
            p_list.append(p_parent)
        
        if len(self.child_parent_info[task_name]['children']) > 0:
            p_childs = []
            for child_name in self.child_parent_info[task_name]['children']:
                p_childs += [self.get_p_task(child_name, positive_relationship=True)]
            p_child = np.mean(p_childs)
            p_list.append(p_child)

        return np.mean(p_list)
    
    def calc_probability_alt(self, task_name):
        p_self = self.get_p_task(task_name, positive_relationship=False)
        
        p_parent = self.clip_p(1)
        if self.child_parent_info[task_name]['parent'] is not None:
            p_parent = self.get_p_task(self.child_parent_info[task_name]['parent'], positive_relationship=False)
            
        p = min(p_self, p_parent)
        
        child_p = self.clip_p(0)
        if len(self.child_parent_info[task_name]['children']) > 0:
            p_childs = []
            for child_name in self.child_parent_info[task_name]['children']:
                p_childs += [self.get_p_task(child_name, positive_relationship=True)]
            child_p = np.mean(p_childs)
            
        p = max(p, child_p)
        
        return p
        
    
    def get_p_task(self, task_name, positive_relationship=True):
        success = self.agent_conductor.task_stats['success'].get_task_edma(task_name)
        if success is None:
            success = 0
        if positive_relationship:
            p = success
        else:
            p = 1 - success
        return self.clip_p(p)
    
    def clip_p(self, p):
        return min(max(p, 0.1), 0.9)

    def next_task(self):
        # TODO: change so calc probs of all tasks, normalized, and sample?
        i = self.last_returned_i + 1
        
        attempts = 0
        while True:
            
            i = i % len(self.tasks_list)
            task_name = self.tasks_list[i]
            p = self.calculate_probability(task_name)
            if random.random() < p:
                self.last_returned_i = i
                return task_name
            i += 1
            self.stat_tracker.append_stat(task_name, p)
            
            attempts += 1
            if attempts > 1000:
                assert False, "Could not find a task to return"