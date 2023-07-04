import numpy as np
import random

from llm_curriculum.envs.tasks.task_trees import TaskTreeBuilder
from llm_curriculum.envs.curriculum_manager import InitialStateCurriculumManager

from llm_curriculum.utils.stats import StatsTracker, ExponentialDecayingMovingAverage


class AgentConductor:
    def __init__(
        self,
        env,
        manual_decompose_p=None,
        dense_rew_lowest=False,
        single_task_names=[],
        high_level_task_names=None,
        contained_sequence=False,
        use_language_goals=False,
        dense_rew_tasks=[],
        use_incremental_reward=False,
        initial_state_curriculum_p=0.0,
    ):
        self.env = env  # TODO: unclear what this is doing
        self.manual_decompose_p = manual_decompose_p

        self.single_task_names = single_task_names

        self.high_level_task_names = high_level_task_names
        if len(self.single_task_names) > 0:
            assert self.manual_decompose_p == 1

        self.contained_sequence = contained_sequence
        if self.contained_sequence:
            assert len(self.single_task_names) == 1

        assert not (dense_rew_lowest and (len(dense_rew_tasks) > 0))
        self.dense_rew_lowest = dense_rew_lowest
        self.dense_reward_tasks = dense_rew_tasks

        self.use_incremental_reward = use_incremental_reward
        self.initial_state_curriculum_manager = InitialStateCurriculumManager(
            initial_state_curriculum_p, self
        )

        # logger
        self.logger = None

        # tasks
        self.high_level_task_list = self.init_possible_tasks(env)
        self.task_names = self.get_task_names()
        self.task_idx_dict, self.n_tasks = self.init_oracle_goals()
        self.task_name2obj_dict = self.set_task_name2obj_dict()
        self.init_task_relations(self.task_names)
        self.init_child_proportions(self.task_names)
        if len(dense_rew_tasks) > 0:
            self.init_dense_reward_tasks(dense_rew_tasks)

        # check single tasks are contained within high-level task tree
        if len(self.single_task_names) > 0:
            assert len(self.high_level_task_list) == 1, "only deal with this for now..."
            all_task_names = self.get_task_names()
            assert all(
                [
                    single_task in all_task_names
                    for single_task in self.single_task_names
                ]
            ), "Single task not contained within high-level task"

        # stats
        self.task_stats = {}
        self.task_stats["success"] = StatsTracker(self.task_names)
        self.task_stats["chosen_n"] = StatsTracker(self.task_names)
        self.task_stats["length"] = StatsTracker(self.task_names)

        # language embeddings
        # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        # Or use same as 'Guided pretraining' paper? https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2
        if use_language_goals:
            raise NotImplementedError()

        # init (to be overrided)
        self.active_task = self.high_level_task_list[0]
        # self.active_task_steps_active = 0

        # tree traversal mode
        self.set_tree_traversal_mode("train")

    def init_possible_tasks(self, env):
        for task in self.high_level_task_names:
            if (
                task.name == "open_then_place_in_drawer"
                or "open_then_place_drawer_then_close"
            ):
                assert env.max_steps >= 80
            else:
                assert env.max_steps == 50
        tree_builder = TaskTreeBuilder(
            use_dense_reward_lowest_level=self.dense_rew_lowest,
            use_incremental_reward=self.use_incremental_reward,
        )
        high_level_tasks = tree_builder.build_from_name_list(self.high_level_task_names)
        assert len(high_level_tasks) == 1, "only set for 1 task currently"
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

    def init_task_relations(self, task_names):
        for task_name in task_names:
            task = self.get_task_from_name(task_name)
            task.record_relations()

    def init_child_proportions(self, task_names):
        for task_name in task_names:
            task = self.get_task_from_name(task_name)
            task.record_child_proportions()

    def init_dense_reward_tasks(self, dense_reward_tasks):
        # possible_task_names = self.get_possible_task_names()
        for task_name in dense_reward_tasks:
            assert task_name in self.task_names
            task = self.get_task_from_name(task_name)
            task.set_use_dense_reward(True)

    def get_task_names(self):
        def recursively_get_names(task):
            task_names = [task.name]
            if len(task.subtask_sequence) > 0:
                for subtask in task.subtask_sequence:
                    subtask_names = recursively_get_names(subtask)
                    task_names += subtask_names
            return task_names

        assert len(self.high_level_task_list) == 1
        task_names = []
        for task in self.high_level_task_list:
            task_names += recursively_get_names(task)

        return task_names

    def set_task_name2obj_dict(self):
        assert len(self.high_level_task_list) == 1
        # find task object
        def recursively_search_for_tasks(task):
            name2obj_dict = {task.name: task}
            if len(task.subtask_sequence) > 0:
                for subtask in task.subtask_sequence:
                    name2obj_dict.update(recursively_search_for_tasks(subtask))
            return name2obj_dict

        high_level_task = self.high_level_task_list[0]
        assert (
            high_level_task.next_task is None
        ), "Can't have next task if no parent task."

        name2obj_dict = recursively_search_for_tasks(high_level_task)
        return name2obj_dict

    def get_single_task_names(self):
        return self.single_task_names

    def get_active_single_task_name(self):
        return self.active_single_task_name

    def set_single_task_names(self, single_task_names):
        self.single_task_names = single_task_names

    def set_curriculum_manager(self, curriculum_manager):
        self.curriculum_manager = curriculum_manager

    def set_logger(self, logger):
        self.logger = logger

    def init_oracle_goals(self):
        task_idx_dict = {}
        for i, name in enumerate(self.task_names):
            task_idx_dict[name] = i
        return task_idx_dict, len(self.task_names)

    def set_tree_traversal_mode(self, mode):
        assert mode in ["train", "leaf", "exploit"]
        self.tree_traversal_mode = mode

    def reset(self):
        # reset tasks (i.e. set complete=False)
        self.reset_tasks()
        # reset intial-state curriculum
        self.initial_state_curriculum_manager.episode_reset()
        # if doing single task per episode (or sequenced), sample from self.single_task_names
        if len(self.single_task_names) > 0:
            self.active_single_task_name = np.random.choice(self.single_task_names)
        else:
            self.active_single_task_name = None
        # choose random high-level task from list # TODO: improve
        self.chosen_high_level_task = np.random.choice(self.high_level_task_list)
        # choose active task
        self.active_task = self.decompose_task(self.chosen_high_level_task)

        return self.active_task

    def reset_tasks(self):
        for high_level_task in self.high_level_task_list:
            high_level_task.reset()

    def step(self):
        prev_active_task = self.active_task
        self.active_task = self.step_task_recursive(prev_active_task)
        return self.active_task

    def decompose_task(self, task, force_decompose=False):
        # Never decompose single task or contained sequence
        if task.name == self.active_single_task_name:
            return task
        if not task.complete:
            if len(task.subtask_sequence) > 0:
                if self.decide_decompose(task) or force_decompose:
                    for subtask in task.subtask_sequence:
                        if not subtask.complete:
                            return self.decompose_task(subtask)
        # Don't decompose IF: (i) is complete, OR (ii) has no subtasks, OR (iii) decided not to decompose
        return task

    def decide_decompose(self, task):
        """
        - Train mode: decompose based on curriculum, or manually defined probability
        - Leaf mode: always decompose (to the leafs)
        - Exploit mode: decompose if gives higher success rate
        """
        if self.tree_traversal_mode == "train":
            if self.manual_decompose_p is None:
                # curriculum
                decompose_p = self.curriculum_manager.calc_decompose_p(task.name)
            else:
                # manual
                decompose_p = self.manual_decompose_p

            do_decompose = np.random.choice(
                [True, False], p=[decompose_p, 1 - decompose_p]
            )
            if self.logger is not None:
                self.logger.record(f"curriculum/{task.name}_decompose_p", decompose_p)

        elif self.tree_traversal_mode == "leaf":
            do_decompose = True

        elif self.tree_traversal_mode == "exploit":
            do_decompose = self.should_decompose_if_exploiting(task)

        return do_decompose

    def step_task_recursive(self, task):
        # if using single tasks
        if task.name == self.active_single_task_name:
            return self.step_single_task(task)
        # normal tree-traversal logic
        else:
            active_task = self.step_task_tree(task)
            # check for initial state curriculum, only if not doing single task learning
            active_task = self.initial_state_curriculum_manager.step(active_task)
            return active_task

    def step_single_task(self, task):
        # contained sequence logic
        if self.contained_sequence:
            if task.complete and task.check_next_task_exists():
                # jump to next task in sequence
                self.active_single_task_name = task.next_task.name
                return task.next_task
        # single task logic
        return task

    def step_task_tree(self, task):
        """
        Once gone down (via decompose_task), only ever come back up 1 level if finished a sub-task sequence
        """
        if task.complete:

            def step_sub_task(task):
                assert (
                    task.complete
                ), f"Task {task.name} is not complete!"  # TODO: shouldn't assume that child sequence completes parent!
                if task.check_next_task_exists():
                    assert task.parent_task is not None
                    # If completed task, parent exists, and next exist - replan from next_task
                    return self.decompose_task(task.next_task)
                else:
                    if self.chosen_high_level_task.complete:
                        # if high_level completed then just return self (nothing else to do!)
                        return task
                    else:
                        assert (
                            task.parent_task is not None
                        )  # if self is complete and high-level is not, then task must have parent - more work to do!
                        # If no next but has parent, then go up to parent
                        return step_sub_task(task.parent_task)

            return step_sub_task(task)
        else:
            # If task not complete, then keep trying!
            return task

    def get_oracle_action(self, state, task):
        # assert self.chosen_high_level_task != 'grasp_cube', "oracle actions don't work well here!!"
        direction_act, gripper_act = task.get_oracle_action(state)
        return FetchAction(self.env, direction_act, gripper_act).get_action()

    def record_task_chosen_stat(self, task, n_steps):
        self.task_stats["chosen_n"].append_stat(task.name, 1)
        self.task_stats["length"].append_stat(task.name, n_steps)

    def record_task_success_stat(self, task, is_success):
        self.task_stats["success"].append_stat(task.name, is_success)

    def get_task_epoch_success_rate(self, task_name):
        success_stats = self.task_stats["success"].get_agg_stats(lambda x: np.mean(x))
        assert False
        return success_stats["epoch"][task_name]

    def get_stats(self):
        stats = {}
        stats["success"] = self.task_stats["success"].get_agg_stats(
            lambda x: np.mean(x)
        )
        stats["chosen_n"] = self.task_stats["chosen_n"].get_agg_stats(lambda x: len(x))
        stats["length"] = self.task_stats["length"].get_agg_stats(lambda x: np.mean(x))
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

    def get_active_task_reward(self, state):
        assert len(self.single_task_names) == 1
        assert self.active_task.name == self.single_task_names[0]
        _, reward = self.active_task.check_success_reward(state)
        return reward

    def get_task_name2obj_dict(self):
        return self.task_name2obj_dict

    def get_task_from_name(self, task_name):
        return self.task_name2obj_dict[task_name]

    def get_possible_task_names(self):
        if len(self.single_task_names) > 0:
            if self.contained_sequence:
                assert len(self.single_task_names) == 1
                task_names = []
                task = self.get_task_from_name(self.single_task_names[0])
                while task.next_task is not None:
                    task_names.append(task.name)
                    task = task.next_task
                task_names.append(task.name)
                return task_names
            else:
                return self.single_task_names
        else:
            return self.get_task_names()

    ###############
    # Exploit mode
    ###############

    def should_decompose_if_exploiting(self, task):
        """
        Gets the maximum possible success that could be obtained via decomposition, and compares to the success of the task itself.

        TODO:
        - Use edma or just the epochs success rate??
        """
        if len(task.subtask_sequence) > 0:
            task_success = self.task_stats["success"].get_task_edma(task.name)
            sequence_success = self.get_sequence_exploit_success(task.subtask_sequence)
            return sequence_success > task_success
        else:
            return False

    def get_sequence_exploit_success(self, sequence):
        """
        Returns max possible success of a sequence of tasks, allowing for decomposition.
        """
        sequence_success = 1
        for task in sequence:
            task_success = self.task_stats["success"].get_task_edma(task.name)
            subtask_sequence = task.subtask_sequence
            if len(subtask_sequence) > 0:
                task_subtask_seq_success = self.get_sequence_exploit_success(
                    subtask_sequence
                )
                task_success = max(task_success, task_subtask_seq_success)
            sequence_success *= task_success
        return sequence_success


class FetchAction:
    def __init__(self, env, direction, gripper_open):
        self.env = env
        self.action_direction = np.clip(direction * 10, -1.0, 1.0)
        if gripper_open:
            self.action_gripper = np.array([1])
        else:
            self.action_gripper = np.array([-1])

        self.env_action = np.concatenate(
            [self.action_direction, self.action_gripper]
        ).astype("float32")
        assert self.env_action in self.env.action_space

    def get_action(self):
        return self.env_action
