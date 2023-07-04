import numpy as np

from llm_curriculum.envs.tasks.state_parsers import CoreStateParser, DrawerStateParser

# TODO: add velocity penalty??? Arm should be still when task complete??


valid_tasks = [
    "move_cube_to_target",
    "pick_up_cube",
    "move_gripper_to_cube",
    "grasp_cube",
    # "open_gripper",
    "cube_between_grippers",
    "close_gripper_cube",
    "lift_cube",
    "place_cube_at_target",
    "move_cube_towards_target_grasp",  # "move_gripper_to_target_grasp",
]
# TODO: not in use anymore... (comment but still use to copy task names quickly)


class Task:
    """
    WARNING: Task should only exist within a single rollout (i.e. re-initialize each time the nvironmnet is reset!!)
    - Note: very important that final child in sequence has identical success condition to parent (child MUST complete parent)
    """

    def __init__(
        self,
        parent_task=None,
        level=0,
        use_dense_reward_lowest_level=False,
        use_incremental_reward=False,
        complete_thresh=3,
    ):
        self.parent_task = parent_task
        self.level = level
        self.use_dense_reward_lowest_level = use_dense_reward_lowest_level
        self.complete_thresh = complete_thresh
        self.use_incremental_reward = use_incremental_reward

        self.next_task = None

        # completion tracking - within episode stats!!
        self.complete = False
        self.success_count = 0

    def set_state_parser_full_tree(self, env_type):
        assert env_type in ["core", "drawer"]
        if env_type == "core":
            self.state_parser = CoreStateParser()
        elif env_type == "drawer":
            self.state_parser = DrawerStateParser()
        # set children
        if len(self.subtask_sequence) > 0:
            for subtask in self.subtask_sequence:
                subtask.set_state_parser_full_tree(env_type)

    def set_subtask_sequence(self, subtask_sequence):
        # init subtasks
        self.subtask_sequence = subtask_sequence
        # set next tasks
        if len(self.subtask_sequence) > 1:
            for i in range(len(self.subtask_sequence) - 1):
                self.subtask_sequence[i].set_next_task(self.subtask_sequence[i + 1])
        # set dense rewards
        if len(self.subtask_sequence) == 0 and self.use_dense_reward_lowest_level:
            self.use_dense_reward = True
        else:
            self.use_dense_reward = False

    def get_str_description(self):
        return self.str_description

    def to_string_full_tree(self, indent=0):
        # Print with indent
        print("  " * indent + self.name)
        if len(self.subtask_sequence) > 0:
            for subtask in self.subtask_sequence:
                subtask.to_string_full_tree(indent + 1)

    def get_leaf_task_sequence(self):
        def _get_leaf_task_sequence_recursive(task):
            leaf_seq = []
            if len(task.subtask_sequence) > 0:
                for subtask in task.subtask_sequence:
                    subtask_seq = _get_leaf_task_sequence_recursive(subtask)
                    leaf_seq += subtask_seq
            else:
                leaf_seq += [task]
            return leaf_seq

        leaf_seq = []
        for subtask in self.subtask_sequence:
            leaf_seq += _get_leaf_task_sequence_recursive(subtask)
        return leaf_seq

    def set_next_task(self, next_task):
        self.next_task = next_task

    def get_next_task(self):
        return self.next_task

    def check_next_task_exists(self):
        return self.next_task is not None

    def active_task_check_and_set_success(self, current_state):
        # Check if task is complete
        success, reward = self.check_success_reward(current_state)
        if success:
            # If so, set to complete
            self.set_complete(success)

            # check parents
            def check_parent_success(task, current_state):
                if task.next_task is None and task.parent_task is not None:
                    parent_success, _ = task.parent_task.check_success_reward(
                        current_state
                    )
                    task.parent_task.set_complete(parent_success)
                    check_parent_success(task.parent_task, current_state)

            check_parent_success(self, current_state)

        # Check if subtasks complete
        self.check_and_set_subtasks_complete(self, current_state)

        return success, reward

    def check_and_set_subtasks_complete(self, task, current_state):
        if len(task.subtask_sequence) > 0:
            for subtask in task.subtask_sequence:
                if subtask.complete:
                    continue
                else:
                    # check complete
                    subtask_success, _ = subtask.check_success_reward(current_state)
                    subtask.set_complete(subtask_success)
                    subtask.check_and_set_subtasks_complete(
                        subtask, current_state
                    )  # TODO: could be weird scenarios where task complete but subtasks arent??
                    return

    def check_success_reward(self, current_state):
        success, reward = self._check_success_reward(current_state)
        if self.use_incremental_reward:
            reward += self.get_incremental_reward(self)
            reward = np.clip(reward, -1, 0)
        return success, reward

    def _check_success_reward(self, current_state):
        raise NotImplementedError("Subclass must implement abstract method")

    def get_oracle_action(self, state):
        assert len(self.subtask_sequence) > 0
        # Find non-complete subtask, and get its oracle action
        for task in self.subtask_sequence:
            if not task.complete:
                return task.get_oracle_action(state)
        return task.get_oracle_action(state)

    def get_incremental_reward(self, task, inc_rew_mag=0.1):
        """
        Currently just add 0.1 to reward for each child complete
        - and recursively go down tree

        TODO: instead do inc_rew = 0.1 * subtask reward?

        """
        inc_rew = 0
        if len(task.subtask_sequence) > 0:
            assert len(task.subtask_sequence) <= 3
            for subtask in task.subtask_sequence:
                if subtask.success_count > 0:
                    inc_rew += inc_rew_mag
                else:
                    subtask_n_childs = len(subtask.subtask_sequence)
                    if subtask_n_childs > 0:
                        child_inc_rew = self.get_incremental_reward(subtask)
                        inc_rew += child_inc_rew / subtask_n_childs
                    break
        return inc_rew

    def record_relations(self):
        """
        Records any relations and their distance from task
        """
        # parents
        def set_parent_relations(task, distance=1):
            parents_dict = {}
            parent_task = task.parent_task
            if parent_task is not None:
                parents_dict[parent_task.name] = distance
                parents_dict.update(
                    set_parent_relations(parent_task, distance=distance + 1)
                )
            return parents_dict

        parents_dict = set_parent_relations(self, distance=1)

        # children
        def set_child_relations(task, distance=1):
            child_dict = {}
            if len(task.subtask_sequence) > 0:
                for child_task in task.subtask_sequence:
                    child_dict[child_task.name] = distance
                    child_dict.update(
                        set_child_relations(child_task, distance=distance + 1)
                    )
            return child_dict

        child_dict = set_child_relations(self, distance=1)
        # set relations
        self.relations = {"parents": parents_dict, "children": child_dict}

    def record_child_proportions(self):
        """
        Records what percent of the task each child completes
        """
        # children
        def set_child_proportions(task, task_prop=1.0):
            assert task_prop <= 1.0
            child_props = {}
            if len(task.subtask_sequence) > 0:
                for child_task in task.subtask_sequence:
                    child_prop = (1.0 / len(task.subtask_sequence)) * task_prop
                    child_props[child_task.name] = child_prop
                    child_props.update(
                        set_child_proportions(child_task, task_prop=child_prop)
                    )
            return child_props

        self.child_propostions = set_child_proportions(self, task_prop=1.0)

    def get_relations(self):
        return self.relations

    def get_child_proportions(self):
        return self.child_propostions

    def binary_reward(self, success):
        if success:
            return 0.0
        else:
            return -1.0

    # within-episode stat setting
    def set_complete(self, is_complete):
        if is_complete:
            self.success_count += 1
            if self.success_count >= self.complete_thresh:
                self.complete = True

    def set_use_dense_reward(self, use_dense_reward):
        self.use_dense_reward = use_dense_reward

    def reset(self):
        self.complete = False
        self.success_count = 0
        if len(self.subtask_sequence) > 0:
            for subtask in self.subtask_sequence:
                subtask.reset()


"""
FetchPickAndPlace
 
"""


class MoveCubeToTargetTask(Task):
    """
    The parent task for the FetchPickPlace environment
    """

    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "move_cube_to_target"
        self.str_description = "Move cube to target"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_at_target(current_state)
        reward = self.binary_reward(success)
        return success, reward


class PickUpCubeTask(Task):
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "pick_up_cube"
        self.str_description = "Pick up cube"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_lifted(current_state)
        reward = self.binary_reward(success)
        return success, reward


class MoveGripperToCubeTask(Task):
    """
    TODO: refactor as MoveGripperToObjectTask - takes the object as input (so can deal with different objects...)
    """

    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "move_gripper_to_cube"
        self.str_description = "Go to cube"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_gripper_above_cube(
            current_state
        )
        if self.use_dense_reward:
            if success:
                reward = 0
            else:
                reward = np.clip(dense_reward * 6, -1, 0)
        else:
            # TODO: make sure revert for MTRL!!
            reward = self.binary_reward(success)
        return success, reward

    def get_oracle_action(self, state):
        # move gripper above cube
        cube_pos = (
            self.state_parser.get_cube_pos(state) + self.state_parser.cube_height_offset
        )
        gripper_pos = self.state_parser.get_gripper_pos(state)
        direction = cube_pos - gripper_pos
        gripper_open = True
        return direction, gripper_open


class GraspCubeTask(Task):
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "grasp_cube"
        self.str_description = "Grasp cube"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_grasped(current_state)
        if self.use_dense_reward:
            reward = dense_reward
        else:
            reward = self.binary_reward(success)
        return success, reward


class LiftCubeTask(Task):
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "lift_cube"
        self.str_description = "Lift cube"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_cube_lifted_and_grasped(
            current_state
        )
        # reward
        if self.use_dense_reward:
            reward = dense_reward
        else:
            reward = self.binary_reward(success)
        return success, reward

    def get_oracle_action(self, state):
        """
        Assumes cube already grasped
        """
        return np.array([0, 0, 1]), False


class PlaceCubeAtTargetTask(Task):
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "place_cube_at_target"
        self.str_description = "Place cube at target"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_at_target(current_state)
        reward = self.binary_reward(success)
        return success, reward


# class MoveGripperToTargetGraspTask(Task):
#     '''
#     # TODO: this doesnt seem to work well at all
#         - learns to keep cube grasped, but doesn't learn to move cube to target...
#     '''
#     def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
#         self.name = "move_gripper_to_target_grasp"
#         self.str_description = "Move gripper to target while grasping the cube"
#         self.subtask_cls_seq = []

#         super().__init__(parent_task, self.subtask_cls_seq, level, use_dense_reward_lowest_level)

#     def check_success_reward(self, current_state):
#         cube_success, cube_dense_reward = self.state_parser.check_cube_at_target(current_state)
#         grasp_success, grasp_dense_reward = self.state_parser.check_grasped(current_state)
#         success = cube_success
#         # reward
#         if cube_success:
#             reward = self.binary_reward(cube_success)
#         else:
#             grasp_reward = self.binary_reward(grasp_success)
#             cube_reward = self.binary_reward(cube_success)
#             reward = (grasp_reward + cube_reward) / 2
#         return success, reward

#     def get_oracle_action(self, state):
#         # move gripper to target while holding cube
#         target_pos = self.state_parser.get_target_pos(state)
#         gripper_pos = self.state_parser.get_gripper_pos(state)
#         direction = target_pos - gripper_pos
#         gripper_open = False
#         return direction, gripper_open


class MoveCubeTowardsTargetGraspTask(Task):
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "move_cube_towards_target_grasp"
        self.str_description = "Move cube to target while grasping"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        if self.use_dense_reward:
            success, dense_reward = self.state_parser.check_cube_at_target(
                current_state
            )
            if success:
                reward = 0
            else:
                reward = np.clip(dense_reward * 3, -1, 0)
        else:
            # # old shaped sparse reward:
            # success, dense_reward = self.state_parser.check_cube_moving_to_target(current_state)
            # reward = self.binary_reward(success)
            # standard sparse reward:
            success, _ = self.state_parser.check_cube_at_target(current_state)
            reward = self.binary_reward(success)
        return success, reward

    def get_oracle_action(self, state):
        # move gripper to target while holding cube
        target_pos = self.state_parser.get_target_pos(state)
        gripper_pos = self.state_parser.get_gripper_pos(state)
        direction = target_pos - gripper_pos
        gripper_open = False
        return direction, gripper_open


# TODO: add these
# class MoveGripperDirection(Task):
# class MoveGripperDirectionGrasp(Task):


class OpenGripperTask(Task):
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "open_gripper"
        self.str_description = "Open gripper"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_gripper_open_stationary(
            current_state
        )
        # reward
        if self.use_dense_reward:
            reward = dense_reward
        else:
            reward = self.binary_reward(success)
        return success, reward

    def get_oracle_action(self, state):
        return np.array([0, 0, 0]), True


class CubeBetweenGripperTask(Task):
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "cube_between_grippers"
        self.str_description = "Put grippers around cube"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_cube_between_grippers_easy(
            current_state
        )
        # reward
        if self.use_dense_reward:
            reward = dense_reward
        else:
            reward = self.binary_reward(success)
        return success, reward

    def get_oracle_action(self, state):
        cube_between_gripper, _ = self.state_parser.check_cube_between_grippers_easy(
            state
        )
        if not cube_between_gripper:
            # Put gripper around cube (assumes gripper above cube and open)
            target_pos = self.state_parser.get_cube_pos(state)
            gripper_pos = self.state_parser.get_gripper_pos(state)
            direction = target_pos - gripper_pos
        else:
            # Do nothing
            direction = np.array([0, 0, 0])
        return direction, True


class CloseGripperCubeTask(Task):
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "close_gripper_cube"
        self.str_description = "Close gripper around cube"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_grasped(current_state)
        # reward
        if self.use_dense_reward:
            reward = dense_reward
        else:
            reward = self.binary_reward(success)
        return success, reward

    def get_oracle_action(self, state):
        # Put gripper around cube
        cube_between_grippers, _ = self.state_parser.check_cube_between_grippers(state)
        if not cube_between_grippers:
            target_pos = self.state_parser.get_cube_pos(state)
            gripper_pos = self.state_parser.get_gripper_pos(state)
            direction = target_pos - gripper_pos
            return direction, True
        # Close gripper
        return np.array([0, 0, 0]), False


# TaskDict = {
#     "move_cube_to_target": MoveCubeToTargetTask(),
#         "pick_up_cube": PickUpCubeTask(),
#             "move_gripper_to_cube": MoveGripperToCubeTask(),
#             "grasp_cube": GraspCubeTask(),
#                 # "open_gripper": OpenGripperTask(), # TODO: add this
#                 # "cube_between_fingers": CubeBetweenFingersTask(),
#                 # "close_gripper_cube": CloseGripperCubeTask(),
#             "lift_cube": LiftCubeTask(),
#         "place_cube_at_target": PlaceCubeAtTargetTask(),
#             "move_gripper_to_target_grasp": MoveGripperToTargetGraspTask(),
# }


# assert all([task_name in valid_tasks for task_name in TaskDict.keys()])


"""
FetchReach - Need a major revamp to get this to work!!!!
"""

# class MoveGripperToTargetTask(Task):
#     def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
#         self.name = "move_gripper_to_target"
#         self.str_description = "Move gripper to target"
#         self.subtask_cls_seq = [MoveGripperDirectionTask]

#         super().__init__(parent_task, self.subtask_cls_seq, level, use_dense_reward_lowest_level)

#     def check_success_reward(self, current_state):
#         success, dense_reward = self.state_parser.check_gripper_at_target(current_state)
#         if self.use_dense_reward_lowest_level:
#             reward = dense_reward
#         else:
#             reward = self.binary_reward(success)
#         return success, reward


# class MoveGripperDirectionTask(Task):
#     # TODO:
#         # - Doesn't work w/ getting string description
#         # - Doesn't work with one-hot goal setup (yet)
#         # - SHould be its own 'multi-option task' class.....
#         # - Need to implement 'next task'... (current sub-task sequence setup doesn't work here...)
#         # - Reset required if using multiple times?????
#         # - Doesn't scale to scenario where such a multi-option task has its own subtasks (e.g. pick red/blue/green cube) - will need a revamp regardless!!!

#         # Main differences: (i) multiple options, (ii) next task not known in advance, (iii) same task object being reused as multiple tasks

#     def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
#         self.name = "move_gripper_direction"
#         # self.str_description = f"Move gripper {direction}"
#         self.subtask_cls_seq = []

#         self.possible_directions = ["Up", "Down", "Left", "Right", "Forward", "Backward"]
#         self.n_variations = len(self.possible_directions)

#         super().__init__(parent_task, self.subtask_cls_seq, level, use_dense_reward_lowest_level)

#         # TODO: set next task here????????????

#     def check_success_reward(self, current_state):
#         success, dense_reward = self.state_parser.check_gripper_direction(current_state)
#         if self.use_dense_reward_lowest_level:
#             reward = dense_reward
#         else:
#             reward = self.binary_reward(success)
#         return success, reward

#     def get_oracle_action(self, state):
#         gripper_pos = self.state_parser.get_gripper_pos(state)
#         target_pos = self.state_parser.get_target_pos(state)
#         direction = target_pos - gripper_pos
#         gripper_open = False
#         return direction, gripper_open

#     def get_next_task(self):
#         return MoveGripperDirectionTask()

#     def check_next_task_exists(self):
#         return True

#     def get_str_description(self):
#         return f"Move gripper {direction}"
