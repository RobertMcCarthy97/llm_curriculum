import numpy as np

class FetchPickPlaceStateParser():
    def __init__(self):
        self.gripper_pos_i = 0
        self.cube_pos_i = 3
        self.right_gripper_i = 9
        self.left_gripper_i = 10
        self.target_i = 25
        
        self.cube_width = 0.025
        self.table_height = 0.425
        
        self.cube_height_offset = np.array([0, 0, 0.06])
        self.gripper_closed_cube_thresh = 0.05 # distance between grippers when grasping cube
        self.cube_lifted_thresh = (self.table_height + 2.5*self.cube_width)
        self.gripper_cube_dist_thresh = (self.cube_width / 2)
        self.gripper_open_thresh = 1.5 * self.cube_width
        self.cube_target_dist_threshold = 0.05
        self.gripper_target_dist_threshold = 0.05
        
        self.scale_cube_between = 10
        self.scale_cube_lifted = 10
        
        
    def get_gripper_pos(self, state):
        return state[self.gripper_pos_i:self.gripper_pos_i+3]
    
    def get_cube_pos(self, state):
        if state.shape[0] < 28:
            assert False
        return state[self.cube_pos_i:self.cube_pos_i+3]
    
    def get_target_pos(self, state):
        return state[self.target_i:self.target_i+3]
    
    def get_distance_grippers(self, state):
        return state[self.right_gripper_i] + state[self.left_gripper_i]
    
    def check_gripper_open(self, state):
        thresh = self.gripper_open_thresh
        distance_between_grippers = self.get_distance_grippers(state)
        success = distance_between_grippers > self.gripper_open_thresh
        dist_clipped  = np.clip(distance_between_grippers, 0, thresh)
        reward = np.clip(-(0.1 - dist_clipped), -1, -thresh)
        return success, reward
    
    def check_cube_between_grippers(self, state):
        thresh = self.gripper_cube_dist_thresh
        dist_gripper_cube = np.linalg.norm(self.get_gripper_pos(state) - self.get_cube_pos(state))
        success = dist_gripper_cube < self.gripper_cube_dist_thresh
        reward = np.clip(-self.scale_cube_between*dist_gripper_cube, -1, 0)
        return success, reward
    
    def check_grippers_closed_cube_width(self, state):
        thresh = self.gripper_closed_cube_thresh
        distance_between_grippers = self.get_distance_grippers(state)
        success = distance_between_grippers <= self.gripper_closed_cube_thresh
        reward = np.clip(-distance_between_grippers, -1, -thresh)
        return success, reward
    
    def check_cube_lifted(self, state):
        thresh = self.cube_lifted_thresh
        cube_z = self.get_cube_pos(state)[2]
        success = cube_z > thresh
        reward = np.clip(-self.scale_cube_lifted*(thresh - cube_z), -1, 0)
        return success, reward
    
    def check_cube_at_target(self, state):
        cube_pos = self.get_cube_pos(state)
        target_pos = self.get_target_pos(state)
        dist = self.distance(cube_pos, target_pos)
        thresh = self.cube_target_dist_threshold
        success = dist < thresh
        reward = np.clip(-dist, -1, 0)
        return success, reward
    
    def check_grasped(self, state):
        cube_between_success, cube_between_rew = self.check_cube_between_grippers(state)
        grip_closed_cube_success, grip_closed_cube_rew = self.check_grippers_closed_cube_width(state)
        success = cube_between_success and grip_closed_cube_success
        reward = np.clip(cube_between_rew + grip_closed_cube_rew, -1, 0)
        return success, reward
    
    def check_cube_lifted_and_grasped(self, state):
        cube_lifted_success, cube_lifted_rew = self.check_cube_lifted(state)
        grasped_success, grasped_rew = self.check_grasped(state)
        success = cube_lifted_success and grasped_success
        reward = np.clip(cube_lifted_rew + grasped_rew, -1, 0)
        return success, reward
    
    def check_cube_at_target_and_grasped(self, state):
        # TODO: scale??
        cube_at_target_success, cube_at_target_rew = self.check_cube_at_target(state)
        cube_grasped_success, cube_grasped_rew = self.check_grasped(state)
        success = cube_at_target_success and cube_grasped_success
        reward = np.clip(cube_at_target_rew + cube_grasped_rew, -1, 0)
        return success, reward
    
    def check_gripper_above_cube(self, state):
        gripper_pos = self.get_gripper_pos(state)
        cube_pos = self.get_cube_pos(state) + self.cube_height_offset
        dist = self.distance(gripper_pos, cube_pos)
        success = (dist < self.gripper_cube_dist_thresh)
        # reward
        # encourage gripper above cube
        reward = np.clip(-dist, -1.0, 0.0)
        # # also encourage open grippers
        # distance_between_grippers = self.state_parser.get_distance_grippers(state)
        # dist_grippers_rew = np.clip(distance_between_grippers - 0.1, -1.0, 0.0)
        return success, reward
    
    def distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)
        


valid_tasks = ['move_cube_to_target',
                'pick_up_cube',
                    'move_gripper_to_cube',
                    "grasp_cube",
                        "open_gripper",
                        "cube_between_fingers",
                        "close_gripper_cube",
                    "lift_cube",
                "place_cube_at_target",
                    "move_gripper_to_target_grasp",
            ]


class Task():
    '''
    WARNING: Task should only exist within a single rollout (i.e. re-initialize each time the nvironmnet is reset!!)
    '''
    def __init__(self, parent_task=None, subtask_cls_seq=[], level=0, use_dense_reward_lowest_level=False, complete_thresh=3):
        assert self.name in valid_tasks
        self.parent_task = parent_task
        self.level = level
        self.next_task = None
        self.use_dense_reward_lowest_level = use_dense_reward_lowest_level
        self.complete_thresh = complete_thresh         
        
        # init subtasks
        self.subtask_sequence = []
        if len(subtask_cls_seq) > 0:
            # init sequence
            for subtask_cls in subtask_cls_seq:
                subtask = subtask_cls(parent_task=self, level=level+1, use_dense_reward_lowest_level=use_dense_reward_lowest_level)
                self.subtask_sequence.append(subtask)
            # set next tasks
            for i in range(len(self.subtask_sequence)-1):
                self.subtask_sequence[i].set_next_task(self.subtask_sequence[i+1])
        
        # state parser
        self.state_parser = FetchPickPlaceStateParser()
        
        # completion tracking - within episode stats!!
        self.complete = False
        self.success_count = 0
    
    def get_str_description(self):
        return self.str_description
        
    def to_string_full_tree(self, indent=0):
        # Print with indent
        print("  "*indent + self.name)
        if len(self.subtask_sequence) > 0:
            for subtask in self.subtask_sequence:
                subtask.to_string_full_tree(indent+1)
                
    def set_next_task(self, next_task):
        self.next_task = next_task
        
    def active_task_check_and_set_success(self, current_state):
        # Check if task is complete
        success, reward = self.check_success_reward(current_state)
        if success:
            # If so, set to complete
            self.set_complete(success)
            if self.next_task is None and self.parent_task is not None:
                # check parent complete and set (if end of subtask sequence complete, parent likely complete)
                parent_success, _ = self.parent_task.check_success_reward(current_state)
                self.parent_task.set_complete(parent_success)
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
                    subtask.check_and_set_subtasks_complete(subtask, current_state) # TODO: could be weird scenarios where task complete but subtasks arent??
                    return
                    
    
    def check_success_reward(self, current_state):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def get_oracle_action(self, state):
        assert len(self.subtask_sequence) > 0
        # Find non-complete subtask, and get its oracle action
        for task in self.subtask_sequence:
            if not task.complete:
                return task.get_oracle_action(state)
        return task.get_oracle_action(state)
        
    def get_incremental_reward(self):
        def recursive_incremental_reward(task, lower_level_perc_complete=1.0):
            if task.parent_task is None:
                assert self.next_task is None
                return lower_level_perc_complete
            else:
                n_tasks = len(task.parent_task.subtask_sequence)
                task_pos = task.parent_task.subtask_sequence.index(self)
                current_level_perc_complete = task_pos / n_tasks
                return recursive_incremental_reward(task.parent_task, current_level_perc_complete)
            
        perc_complete = recursive_incremental_reward(self)
        return perc_complete
    
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
                
    def reset(self):
        self.complete = False
        self.success_count = 0
        if len(self.subtask_sequence) > 0:
            for subtask in self.subtask_sequence:
                subtask.reset()
 
 
class MoveCubeToTargetTask(Task):
    '''
    The parent task for the FetchPickPlace environment
    '''
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "move_cube_to_target"
        self.str_description = "Move cube to target"
        self.subtask_cls_seq = [PickUpCubeTask, PlaceCubeAtTargetTask]
        
        super().__init__(parent_task, self.subtask_cls_seq, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_at_target(current_state)
        reward = self.binary_reward(success)
        return success, reward

    
class PickUpCubeTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "pick_up_cube"
        self.str_description = "Pick up cube"
        self.subtask_cls_seq = [MoveGripperToCubeTask, GraspCubeTask, LiftCubeTask]
        
        super().__init__(parent_task, self.subtask_cls_seq, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_lifted(current_state)
        reward = self.binary_reward(success)
        return success, reward

    
class MoveGripperToCubeTask(Task):
    '''
    TODO: refactor as MoveGripperToObjectTask - takes the object as input (so can deal with different objects...)
    '''
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "move_gripper_to_cube"
        self.str_description = "Move gripper to cube"
        self.subtask_cls_seq = []
        
        super().__init__(parent_task, self.subtask_cls_seq, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_gripper_above_cube(current_state)
        if self.use_dense_reward_lowest_level:
            reward = dense_reward
        else:
            reward = self.binary_reward(success)
        return success, reward
    
    def get_oracle_action(self, state):
        # move gripper above cube
        cube_pos = self.state_parser.get_cube_pos(state) + self.state_parser.cube_height_offset
        gripper_pos = self.state_parser.get_gripper_pos(state)
        direction = cube_pos - gripper_pos
        gripper_open = True
        return direction, gripper_open

class GraspCubeTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "grasp_cube"
        self.str_description = "Grasp cube"
        self.subtask_cls_seq = [] # TODO: ["open_gripper", "cube_between_fingers", "close_gripper_cube"]
        
        super().__init__(parent_task, self.subtask_cls_seq, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_grasped(current_state)
        if self.use_dense_reward_lowest_level:
            reward = dense_reward
        else:
            reward = self.binary_reward(success)
        return success, reward
    
    def get_oracle_action(self, state):
        '''
        Assumes the gripper is above the cube
        '''
        # check if grasped
        grasped, _ = self.state_parser.check_grasped(state)
        if not grasped:
            # Open gripper
            is_gripper_open, _ = self.state_parser.check_gripper_open(state)
            if not is_gripper_open:
                return np.array([0, 0, 0]), True
            # Put gripper around cube
            cube_between_grippers, _ = self.state_parser.check_cube_between_grippers(state)
            if not cube_between_grippers:
                target_pos = self.state_parser.get_cube_pos(state)
                gripper_pos = self.state_parser.get_gripper_pos(state)
                direction = target_pos - gripper_pos
                return direction, True
        # Close gripper
        return np.array([0, 0, 0]), False
 
class LiftCubeTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "lift_cube"
        self.str_description = "Lift cube"
        self.subtask_cls_seq = []
        
        super().__init__(parent_task, self.subtask_cls_seq, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_cube_lifted_and_grasped(current_state)
        # reward
        if self.use_dense_reward_lowest_level:
            reward = dense_reward
        else:
            reward = self.binary_reward(success)
        return success, reward
    
    def get_oracle_action(self, state):
        '''
        Assumes cube already grasped
        '''
        return np.array([0, 0, 1]), False
    
class PlaceCubeAtTargetTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "place_cube_at_target"
        self.str_description = "Place cube at target"
        self.subtask_cls_seq = [MoveGripperToTargetGraspTask]
        
        super().__init__(parent_task, self.subtask_cls_seq, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_at_target(current_state)
        reward = self.binary_reward(success)
        return success, reward
    
class MoveGripperToTargetGraspTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "move_gripper_to_target_grasp"
        self.str_description = "Move gripper to target while grasping the cube"
        self.subtask_cls_seq = []
        
        super().__init__(parent_task, self.subtask_cls_seq, level, use_dense_reward_lowest_level)
    
    def check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_cube_at_target_and_grasped(current_state)
        # reward
        if self.use_dense_reward_lowest_level:
            reward = dense_reward
        else:
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
# class OpenGripperTask(Task):
# class CubeBetweenFingersTask(Task):
# class CloseGripperCubeTask(Task):
# class MoveGripperDirection(Task):
# class MoveGripperDirectionGrasp(Task):


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