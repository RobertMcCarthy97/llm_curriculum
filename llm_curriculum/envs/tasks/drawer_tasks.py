import numpy as np

from llm_curriculum.envs.tasks.core_tasks import Task


#######################
# Task classes
#######################


################## Place cube in drawer


class PlaceCubeDrawerTask(Task):
    """
    The parent task for the FetchPickPlace environment
    """

    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "place_cube_drawer"
        self.str_description = "Place cube in drawer"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_in_drawer(current_state)
        reward = self.binary_reward(success)
        return success, reward


class PlaceGraspedCubeDrawerTask(Task):
    """
    The parent task for the FetchPickPlace environment
    """

    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "place_grasped_cube_drawer"
        self.str_description = "Place grasped cube in drawer"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_in_drawer(current_state)
        reward = self.binary_reward(success)
        return success, reward


class MoveCubeOverDrawerTask(Task):
    # TODO: change success condition so checks for over open drawer area only???
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "move_cube_over_drawer"
        self.str_description = "Move cube over drawer"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_cube_over_dynamic_drawer(
            current_state
        )
        if self.use_dense_reward:
            if success:
                reward = 0
            else:
                reward = np.clip(dense_reward, -1, 0)
        else:
            reward = self.binary_reward(success)
        return success, reward

    def get_oracle_action(self, state):
        """
        Assume grasping cube and drawer open
        """
        # move gripper above handle
        drawer_pos = (
            self.state_parser.get_drawer_dynamic_center(state)
            + self.state_parser.over_drawer_height_offset
        )
        gripper_pos = self.state_parser.get_gripper_pos(state)
        direction = drawer_pos - gripper_pos
        gripper_open = False
        return direction, gripper_open


class ReleaseCubeInDrawerTask(Task):
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "release_cube_in_drawer"
        self.str_description = "Release cube in drawer"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_in_drawer(current_state)
        reward = self.binary_reward(success)
        return success, reward

    def get_oracle_action(self, state):
        """
        Assume grasping cube over open drawer
        """
        gripper_open = True
        return np.array([0, 0, 0]), gripper_open


################### Open/clsoe drawer


class OpenDrawerTask(Task):
    """
    The parent task for the FetchPickPlace environment
    """

    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "open_drawer"
        self.str_description = "Open drawer"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, _ = self.state_parser.check_drawer_open(current_state)
        reward = self.binary_reward(success)
        return success, reward


class CloseDrawerTask(Task):
    """
    The parent task for the FetchPickPlace environment
    """

    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "close_drawer"
        self.str_description = "Close drawer"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, _ = self.state_parser.check_drawer_closed(current_state)
        reward = self.binary_reward(success)
        return success, reward


class MoveGripperToDrawerTask(Task):
    """
    TODO: refactor as MoveGripperToObjectTask - takes the object as input (so can deal with different objects...)
    """

    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "move_gripper_to_drawer"
        self.str_description = "Go to drawer"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_gripper_above_handle(
            current_state
        )
        if self.use_dense_reward:
            if success:
                reward = 0
            else:
                reward = np.clip(dense_reward, -1, 0)
        else:
            reward = self.binary_reward(success)
        return success, reward

    def get_oracle_action(self, state):
        # move gripper above handle
        handle_pos = (
            self.state_parser.get_handle_pos(state)
            + self.state_parser.above_handle_height_offset
        )
        gripper_pos = self.state_parser.get_gripper_pos(state)
        direction = handle_pos - gripper_pos
        gripper_open = False
        return direction, gripper_open


class GraspHandleTask(Task):
    # TODO: implement between and close?
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "grasp_handle"
        self.str_description = "Grasp handle"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_handle_grasped(current_state)
        if self.use_dense_reward:
            reward = dense_reward
        else:
            reward = self.binary_reward(success)
        return success, reward

    def get_oracle_action(self, state):
        """
        Assume already above handle
        """
        handle_pos = self.state_parser.get_handle_pos(state)
        gripper_pos = self.state_parser.get_gripper_pos(state)
        direction = handle_pos - gripper_pos
        gripper_open = True
        return direction, gripper_open


class PullHandleToOpenTask(Task):
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "pull_handle_to_open"
        self.str_description = "Pull handle to open drawer"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, _ = self.state_parser.check_drawer_open(current_state)
        reward = self.binary_reward(success)
        return success, reward

    def get_oracle_action(self, state):
        """
        Assumes handle already grasped - move gripper in direction drawer opens
        """
        success, _ = self.state_parser.check_drawer_open(state)
        if success:
            return np.array([0, 0, 0]), False
        else:
            return np.array([0, -1, 0]), False


class PushHandleToCloseTask(Task):
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "push_handle_to_close"
        self.str_description = "Push handle to close drawer"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, _ = self.state_parser.check_drawer_closed(current_state)
        reward = self.binary_reward(success)
        return success, reward

    def get_oracle_action(self, state):
        """
        Assumes handle already grasped - move gripper in direction drawer closes
        """
        success, _ = self.state_parser.check_drawer_closed(state)
        if success:
            return np.array([0, 0, 0]), False
        else:
            return np.array([0, 1, 0]), False


################### Place cube ON drawer


class PlaceCubeOnDrawerTopTask(Task):
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "place_cube_on_drawer_top"
        self.str_description = "Place cube on drawer top"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_on_drawer_top(current_state)
        reward = self.binary_reward(success)
        return success, reward


class PlaceGraspedCubeOnDrawerTopTask(Task):
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "place_grasped_cube_on_drawer_top"
        self.str_description = "Place grasped cube on drawer top"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_on_drawer_top(current_state)
        reward = self.binary_reward(success)
        return success, reward


class MoveCubeOverDrawerTopTask(Task):
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "move_cube_over_drawer_top"
        self.str_description = "Move cube over drawer top"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_cube_over_drawer_top(
            current_state
        )
        if self.use_dense_reward:
            if success:
                reward = 0
            else:
                reward = np.clip(dense_reward, -1, 0)
        else:
            reward = self.binary_reward(success)
        return success, reward

    def get_oracle_action(self, state):
        """
        Assumes cube grasped
        """
        gripper_pos = self.state_parser.get_gripper_pos(state)
        above_pos = self.state_parser.get_static_above_rect_volume(state).get_center()
        direction = above_pos - gripper_pos
        gripper_open = False
        return direction, gripper_open


class ReleaseCubeOnDrawerTopTask(Task):
    def __init__(
        self, parent_task=None, level=0, use_dense_reward_lowest_level=False, **kwargs
    ):
        self.name = "release_cube_on_drawer_top"
        self.str_description = "Release cube on drawer top"

        super().__init__(parent_task, level, use_dense_reward_lowest_level, **kwargs)

    def _check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_on_drawer_top(current_state)
        reward = self.binary_reward(success)
        return success, reward

    def get_oracle_action(self, state):
        """
        Assumes cube already grasped over drawer
        """
        gripper_open = True
        return np.array([0, 0, 0]), gripper_open
