import random
import numpy as np

from llm_curriculum.utils.stats import StatsTracker


class CurriculumManager:
    def __init__(
        self, tasks_list, agent_conductor, child_p_strat="mean", p_combined_strat="mean"
    ):
        self.tasks_list = tasks_list
        self.agent_conductor = agent_conductor
        self.child_p_strat = child_p_strat
        self.p_combined_strat = p_combined_strat

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
    def __init__(self, tasks_list, agent_conductor, child_p_strat="mean"):
        super().__init__(tasks_list, agent_conductor, child_p_strat)

    def next_task(self):
        i = self.last_returned_i + 1
        i = i % len(self.tasks_list)
        chosen_task = self.tasks_list[i]
        self.last_returned_i = i
        self.stat_tracker.append_stat(chosen_task, 1)
        return chosen_task


class SeperateEpisodesCM(CurriculumManager):
    def __init__(self, tasks_list, agent_conductor, child_p_strat="mean"):
        super().__init__(tasks_list, agent_conductor, child_p_strat)

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

    def calc_decompose_p(self, task_name):
        """
        Decide probability of whether to decompose a task

        Child_p is just the average child_p (# TODO: improve)

        Flaws:
        - If self bad (0.0) and child good (1.0), still decomposes 50% of time. Should focus more on self here!! (actually, this is fine because child data can be useful?)
        - If self good (1.0) and child bad (0.0), still decomposes 50% of time. Mostly care about higher level tasks, so should focus more on self here?

        # TODO:
        - Take max p?
        - Or just add more weight to the higher p??
        """

        # p based on tasks own success rate
        p_self = self.get_p_task(
            task_name, positive_relationship=True
        )  # (better self is, more likely to stick with self)

        # p of sticking with task based on children success rates
        if len(self.child_parent_info[task_name]["children"]) > 0:

            if self.child_p_strat == "mean":
                # TODO: issue - gives same weight to each child...
                p_childs = []
                for child_name in self.child_parent_info[task_name]["children"]:
                    p_childs += [
                        self.get_p_task(child_name, positive_relationship=True)
                    ]  # (better child is, more likely to stick with self)
                p_child = np.mean(p_childs)

            elif self.child_p_strat == "last":
                # TODO: what if last has good success rate but only been run very few times??
                last_child_name = self.task_last_childs[task_name]
                p_child = self.get_p_task(last_child_name, positive_relationship=True)
                assert False, "not thought through yet..."

            elif self.child_p_strat == "sequenced":
                task = self.agent_conductor.get_task_from_name(task_name)
                child_seq_success = self.agent_conductor.get_sequence_exploit_success(
                    task.subtask_sequence
                )
                p_child = self.clip_p(child_seq_success)

            elif self.child_p_strat == "sequenced_direct_children":
                p_child = 1
                task = self.agent_conductor.get_task_from_name(task_name)
                for child in self.subtask_sequence:
                    p_child *= self.get_p_task(child.name, positive_relationship=True)
                assert False, "not tested"

            else:
                raise NotImplementedError

        # combined p
        if self.p_combined_strat == "mean":
            p = np.mean([p_self, p_child])
            # p is probability of sticking with task, so decompose_p is 1-p
            decompose_p = 1 - p

        elif self.p_combined_strat == "proportional":
            decompose_p = (1 - p_self) * (1 - p_child)

        elif self.p_combined_strat == "mean_parent_clip":
            if p_self > p_child:
                decompose_p = 1 - p_self
            else:
                decompose_p = 1 - np.mean([p_self, p_child])

        return decompose_p


class InitialStateCurriculumManager:
    """
    Super basic and unprincipled approach to initial state curriculum.

    - Decides wheter to do intial state curriculum for a task
    - Decides which intial-state child to use
    - Returns parent once intial-state reached
    - inital_state_curriculum_p decides whether to do the curriculum
    - only allowed apply intial-state curriculum once to a single task within a single episode

    TODO:
    - Exploit mode (take tree path most likely to reach desired intial state)
    - Use success-rate based p (no point doing curriculum if children are bad, or parent better than children)
    - COmplete revamp
    - More principled choice of child
    """

    def __init__(self, initial_state_curriculum_p, agent_conductor):
        self.initial_state_curriculum_p = initial_state_curriculum_p
        self.agent_conductor = agent_conductor
        self.episode_reset()

    def episode_reset(self):
        self.curriculum_reset()
        self.tasks_to_not_apply_curric_again_to = []

    def curriculum_reset(self):
        self.task_requiring_curriculum = None
        self.completing_reset_task = None
        self.curriculum_in_progress = False

    def step(self, active_task):
        if self.curriculum_in_progress:
            # if reached init state, then deploy parent
            if self.completing_reset_task.success_count > 0:
                og_task = self.task_requiring_curriculum
                self.curriculum_reset()
                self.tasks_to_not_apply_curric_again_to += [og_task.name]
                return og_task
        else:
            task_requiring_curriculum = active_task
            # check if have already done curriculum for this task
            already_done = (
                task_requiring_curriculum.name
                in self.tasks_to_not_apply_curric_again_to
            )
            if not already_done:
                # decide whether to do curriculum
                (
                    completing_reset_task,
                    _,
                    do_curriculum,
                ) = self.decide_initial_state_curriculum_task(
                    task_requiring_curriculum
                )  # this checks if n_children > 1
                if do_curriculum:
                    self.curriculum_in_progress = True
                    self.task_requiring_curriculum = task_requiring_curriculum
                    self.completing_reset_task = completing_reset_task
                    curriculum_task = self.agent_conductor.decompose_task(
                        task_requiring_curriculum, force_decompose=True
                    )
                    return curriculum_task
                else:
                    # rule out for rest of episode
                    self.tasks_to_not_apply_curric_again_to += [
                        task_requiring_curriculum.name
                    ]
        # else just return original actrive task
        return active_task

    def decide_do_initial_curriculum(self, task):
        assert (
            self.initial_state_curriculum_p == 0.0
        ), "initial-state curriculum not yet robust"
        # decide based on p
        do_curriculum_p = (
            self.initial_state_curriculum_p
        )  # TODO: base this on success rates...
        do_curriculum = np.random.choice(
            [True, False], p=[do_curriculum_p, 1 - do_curriculum_p]
        )
        # decide based on children
        sufficient_children = len(task.subtask_sequence) > 1
        do_curriculum = do_curriculum and sufficient_children
        assert not do_curriculum
        return do_curriculum

    def decide_initial_state_curriculum_task(self, task):
        """
        - Decides whether do curriculum according to initial_state_curriculum_p
        - If so, choose a random leaf task to reset to (excluding first leaf)
        - assumes each leaf is evenly spaced / evenly difficult

        """
        do_curriculum = self.decide_do_initial_curriculum(task)
        if do_curriculum:
            # get leaf nodes
            leaf_sequence = task.get_leaf_task_sequence()
            n_leafs = len(leaf_sequence)
            chosen_i = random.randrange(1, n_leafs)
            init_to_reset_leaf = leaf_sequence[chosen_i]
            completing_reset_leaf = leaf_sequence[chosen_i - 1]
            return completing_reset_leaf, init_to_reset_leaf, True
        else:
            return None, task, False
