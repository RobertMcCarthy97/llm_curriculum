import random
import numpy as np

from llm_curriculum.envs.agent_conductor import StatsTracker


class CurriculumManager:
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

    def init_child_parent_info(self):
        """
        Only records direct children and parents

        """
        tasks_info = {}
        for task_name in self.tasks_list:
            task_info = {"children": [], "parent": None}

            task = self.agent_conductor.get_task_from_name(task_name)

            if len(task.subtask_sequence) > 0:
                for child in task.subtask_sequence:
                    if child.name in self.tasks_list:
                        task_info["children"].append(child.name)

            if task.parent_task is not None:
                if task.parent_task.name in self.tasks_list:
                    task_info["parent"] = task.parent_task.name

            tasks_info[task_name] = task_info

        return tasks_info

    def init_task_last_childs(self):
        task_last_childs = {}
        for task_name in self.tasks_list:
            task = self.agent_conductor.get_task_from_name(task_name)
            if len(task.subtask_sequence) > 0:
                task_last_childs[task_name] = task.subtask_sequence[-1].name
            else:
                task_last_childs[task_name] = None
        return task_last_childs


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
        self.task_last_childs = self.init_task_last_childs()

    def calculate_probability(self, task_name):
        p_list = []

        p_self = self.get_p_task(task_name, positive_relationship=False)
        p_list.append(p_self)

        if self.child_parent_info[task_name]["parent"] is not None:
            p_parent = self.get_p_task(
                self.child_parent_info[task_name]["parent"], positive_relationship=False
            )
            p_list.append(p_parent)

        if len(self.child_parent_info[task_name]["children"]) > 0:
            p_childs = []
            for child_name in self.child_parent_info[task_name]["children"]:
                p_childs += [self.get_p_task(child_name, positive_relationship=True)]
            p_child = np.mean(p_childs)
            p_list.append(p_child)

        return np.mean(p_list)

    def calc_probability_alt(self, task_name):
        p_self = self.get_p_task(task_name, positive_relationship=False)

        p_parent = self.clip_p(1)
        if self.child_parent_info[task_name]["parent"] is not None:
            p_parent = self.get_p_task(
                self.child_parent_info[task_name]["parent"], positive_relationship=False
            )

        p = min(p_self, p_parent)

        child_p = self.clip_p(0)
        if len(self.child_parent_info[task_name]["children"]) > 0:
            p_childs = []
            for child_name in self.child_parent_info[task_name]["children"]:
                p_childs += [self.get_p_task(child_name, positive_relationship=True)]
            child_p = np.mean(p_childs)

        p = max(p, child_p)

        return p

    def get_p_task(self, task_name, positive_relationship=True, use_edma=True):
        if use_edma:
            success_rate = self.agent_conductor.task_stats["success"].get_task_edma(
                task_name
            )  # TODO: make these agent conductor methods
        else:
            success_rate = self.agent_conductor.task_stats[
                "success"
            ].get_task_epoch_agg(task_name)
        if success_rate is None:
            success_rate = 0
        if positive_relationship:
            p = success_rate
        else:
            p = 1 - success_rate
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

    def calc_decompose_p(self, task_name, child_strat="all"):
        """
        Decide probability of whether to decompose a task

        Child_p is just the average child_p (# TODO: improve)

        Flaws:
        - If self bad (0.0) and child good (1.0), still decomposes 50% of time. Should focus more on self here!!
        - If self good (1.0) and child bad (0.0), still decomposes 50% of time. Mostly care about higher level tasks, so should focus more on self here?

        # TODO:
        - Take max p?
        - Or just add more weight to the higher p??
        """
        p_list = []

        # p based on tasks own success rate
        p_self = self.get_p_task(
            task_name, positive_relationship=True
        )  # (better self is, more likely to stick with self)
        p_list.append(p_self)

        # p of sticking with task based on children success rates
        if len(self.child_parent_info[task_name]["children"]) > 0:

            if child_strat == "all":
                p_childs = []
                for child_name in self.child_parent_info[task_name]["children"]:
                    p_childs += [
                        self.get_p_task(child_name, positive_relationship=True)
                    ]  # (better child is, more likely to stick with self)
                p_child = np.mean(p_childs)

            elif child_strat == "last":
                # TODO: what if last has good success rate but only been run very few times??
                last_child_name = self.task_last_childs[task_name]
                p_child = self.get_p_task(last_child_name, positive_relationship=True)
                assert False, "not thought through yet..."

            else:
                raise NotImplementedError

            p_list.append(p_child)

        # combined p
        p = np.mean(p_list)
        # p is probability of sticking with task, so decompose_p is 1-p
        decompose_p = 1 - p
        return decompose_p