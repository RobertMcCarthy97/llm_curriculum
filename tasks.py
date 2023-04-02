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
        self.gripper_closed_cube_thresh = 0.05 # distance between grippers when grasping cube
        self.cube_lifted_thresh = (self.table_height + 2.5*self.cube_width)
        self.gripper_at_cube_thresh = (self.cube_width / 2)
        self.gripper_open_thresh = 1.5 * self.cube_width
        
        self.cube_target_dist_threshold = 0.05
        self.gripper_target_dist_threshold = 0.05
        
        
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
        distance_between_grippers = self.get_distance_grippers(state)
        return distance_between_grippers > self.gripper_open_thresh
    
    def check_cube_between_grippers(self, state):
        dist_gripper_cube = np.linalg.norm(self.get_gripper_pos(state) - self.get_cube_pos(state))
        gripper_at_cube = dist_gripper_cube < self.gripper_at_cube_thresh
        return gripper_at_cube
    
    def check_grasped(self, state):
        gripper_at_cube = self.check_cube_between_grippers(state)
        distance_between_grippers = self.get_distance_grippers(state)
        gripper_closed_cube = distance_between_grippers <= self.gripper_closed_cube_thresh
        return gripper_at_cube and gripper_closed_cube
    
    def check_cube_lifted(self, state):
        lifted = self.get_cube_pos(state)[2] > self.cube_lifted_thresh
        return lifted
    
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
    def __init__(self, parent_task=None, subtask_cls_seq=[], level=0, use_dense_reward_lowest_level=False, complete_thresh=3):
        assert self.name in valid_tasks
        self.parent_task = parent_task
        self.level = level
        self.next_task = None
        self.use_dense_reward_lowest_level = use_dense_reward_lowest_level         
        
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

        # completion tracking
        self.complete = False
        self.success_count = 0
        self.complete_thresh = complete_thresh
        
        # state parser
        self.state_parser = FetchPickPlaceStateParser()
    
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
    
    def set_complete(self, is_complete):
        if is_complete:
            self.success_count += 1
            if self.success_count >= self.complete_thresh:
                self.complete = True
        
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
 
 
class MoveCubeToTargetTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "move_cube_to_target"
        self.str_description = "Move cube to target"
        self.subtask_cls_seq = [PickUpCubeTask, PlaceCubeAtTargetTask]
        
        super().__init__(parent_task, self.subtask_cls_seq, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        cube_pos = self.state_parser.get_cube_pos(current_state)
        target_pos = self.state_parser.get_target_pos(current_state)
        success = (self.state_parser.distance(cube_pos, target_pos) < self.state_parser.cube_target_dist_threshold)
        reward = self.binary_reward(success)
        return success, reward

    
class PickUpCubeTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "pick_up_cube"
        self.str_description = "Pick up cube"
        self.subtask_cls_seq = [MoveGripperToCubeTask, GraspCubeTask, LiftCubeTask]
        
        super().__init__(parent_task, self.subtask_cls_seq, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        grasped = self.state_parser.check_grasped(current_state)
        cube_lifted = self.state_parser.check_cube_lifted(current_state)
        success = (grasped and cube_lifted)
        reward = self.binary_reward(success)
        return success, reward

    
class MoveGripperToCubeTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "move_gripper_to_cube"
        self.str_description = "Move gripper to cube"
        self.subtask_cls_seq = []
        
        super().__init__(parent_task, self.subtask_cls_seq, level, use_dense_reward_lowest_level)
        
        self.height_offset = np.array([0, 0, 0.06])
        
    def check_success_reward(self, current_state):
        gripper_pos = self.state_parser.get_gripper_pos(current_state)
        cube_pos = self.state_parser.get_cube_pos(current_state) + self.height_offset
        distance = self.state_parser.distance(gripper_pos, cube_pos)
        success = (distance < self.state_parser.gripper_at_cube_thresh)
        if self.use_dense_reward_lowest_level:
            # encourage open gripper
            distance_between_grippers = self.state_parser.get_distance_grippers(current_state)
            dist_grippers_rew = np.clip(distance_between_grippers - 0.1, -1.0, 0.0)
            # encourage gripper to be above cube
            dist_rew = np.clip(-distance, -1.0, 0.0)
            reward = np.clip(dist_grippers_rew + dist_rew, -1.0, 0.0)
        else:
            reward = self.binary_reward(success)
        return success, reward
    
    def get_oracle_action(self, state):
        # move gripper above cube
        cube_pos = self.state_parser.get_cube_pos(state) + self.height_offset
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
        # TODO: ensure rewards are always encouraging policy towards a success condition!
        grasped = self.state_parser.check_grasped(current_state)
        success = grasped
        if self.use_dense_reward_lowest_level:
            # check distance to cube
            dist_gripper_cube = np.linalg.norm(self.state_parser.get_gripper_pos(current_state) - self.state_parser.get_cube_pos(current_state)) # TODO: deduplicate this code (see check successes in state_parser)
            dist_gripper_cube_rew = np.clip(-10*dist_gripper_cube, -1.0, 0.0) # scale
            # check gripper around cube
            distance_between_grippers = self.state_parser.get_distance_grippers(current_state)
            dist_grippers_rew = np.clip(-distance_between_grippers, -1.0, -self.state_parser.gripper_closed_cube_thresh)
            # calc rew
            reward = np.clip(dist_gripper_cube_rew + dist_grippers_rew, -1.0, 0.0)
        else:
            reward = self.binary_reward(success)
        return success, reward
    
    def get_oracle_action(self, state):
        '''
        Assumes the gripper is above the cube
        '''
        # check if grasped
        if not self.state_parser.check_grasped(state):
            # Open gripper
            is_gripper_open = self.state_parser.check_gripper_open(state)
            if not is_gripper_open:
                return np.array([0, 0, 0]), True
            # Put gripper around cube
            cube_between_grippers = self.state_parser.check_cube_between_grippers(state)
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
        cube_lifted = self.state_parser.check_cube_lifted(current_state)
        cube_grasped = self.state_parser.check_grasped(current_state)
        success = (cube_lifted and cube_grasped)
        # reward
        if self.use_dense_reward_lowest_level:
            # check gripper around cube
            distance_between_grippers = self.state_parser.get_distance_grippers(current_state)
            dist_grippers_rew = np.clip(-distance_between_grippers, -1.0, -self.state_parser.gripper_closed_cube_thresh)
            # check cube height from table
            cube_height = self.state_parser.cube_lifted_thresh - self.state_parser.get_cube_pos(current_state)[2]
            cube_height_rew = np.clip(-10*cube_height, -1.0, 0.0) # scale
            # calc rew
            reward = np.clip(dist_grippers_rew + cube_height_rew, -1.0, 0.0)
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
        cube_pos = self.state_parser.get_cube_pos(current_state)
        goal_pos = self.state_parser.get_target_pos(current_state)
        success = (self.state_parser.distance(cube_pos, goal_pos) < self.state_parser.cube_target_dist_threshold)
        reward = self.binary_reward(success)
        return success, reward
    
class MoveGripperToTargetGraspTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "move_gripper_to_target_grasp"
        self.str_description = "Move gripper to target"
        self.subtask_cls_seq = []
        
        super().__init__(parent_task, self.subtask_cls_seq, level, use_dense_reward_lowest_level)
        
    # TODO: so much duplication in reward calcs - make cleaner!!
    def check_success_reward(self, current_state):
        # coords
        gripper_pos = self.state_parser.get_gripper_pos(current_state)
        cube_pos = self.state_parser.get_cube_pos(current_state)
        goal_pos = self.state_parser.get_target_pos(current_state)
        # checks
        at_target = (self.state_parser.distance(gripper_pos, goal_pos) < self.state_parser.gripper_target_dist_threshold) # TODO: check cube not gripper
        grasped = self.state_parser.check_grasped(current_state)
        success = (at_target and grasped)
        # reward
        if self.use_dense_reward_lowest_level:
            # check gripper around cube
            distance_between_grippers = self.state_parser.get_distance_grippers(current_state)
            dist_grippers_rew = np.clip(-distance_between_grippers, -1.0, -self.state_parser.gripper_closed_cube_thresh)
            # check gripper at cube
            gripper_cube_dist = self.state_parser.distance(gripper_pos, cube_pos)
            gripper_cube_dist_rew = np.clip(-gripper_cube_dist, -1.0, -self.state_parser.gripper_at_cube_thresh)
            # check cube at target
            dist_cube_target = self.state_parser.distance(cube_pos, goal_pos)
            dist_cube_target_rew = np.clip(-dist_cube_target, -1.0, -self.state_parser.cube_target_dist_threshold)
            # calc rew
            reward = np.clip(dist_grippers_rew + gripper_cube_dist_rew + dist_cube_target_rew, -1.0, 0.0)
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