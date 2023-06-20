import numpy as np

class FetchPickPlaceStateParser():
    '''
    Takes in state observations and returns reward and success for different conditions.

    # TODO: pass in obs space during init also...
    '''
    def __init__(self):
        # position of info in observation vector
        self.gripper_pos_i = 0
        self.cube_pos_i = 3
        self.right_gripper_i = 9
        self.left_gripper_i = 10
        self.target_i = 25 # TODO: this will change if add drawer...
        self.gripper_vel_i = 20
        self.cube_rel_vel_i = 14 # velocity relative to gripper...
        
        # additional env info
        self.cube_width = 0.025
        self.table_height = 0.425
        
        # task related thresholds
        self.cube_height_offset = np.array([0, 0, 0.05]) # 0.05
        self.gripper_closed_cube_thresh = 0.05 # distance between grippers when grasping cube
        self.cube_lifted_thresh = (self.table_height + 2.5*self.cube_width)
        self.cube_between_gripper_dist_thresh = self.cube_width
        self.gripper_open_thresh = 1.8 * self.cube_width
        self.gripper_stationary_thresh = 0.001
        self.cube_moving_vel_threshold = 0.005
        
        self.cube_target_dist_threshold = 0.05
        self.gripper_target_dist_threshold = 0.05
        self.gripper_cube_dist_thresh = 0.05
        
        self.scale_cube_between = 10
        self.scale_cube_lifted = 10
        
        self.velocity_angle_thresh = 45 # degrees
        
        
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
    
    def get_gripper_vel(self, state):
        cartesian_vel = state[self.gripper_vel_i:self.gripper_vel_i+3]
        return cartesian_vel
    
    def get_cube_vel(self, state):
        gripper_vel = self.get_gripper_vel(state)
        cube_relative_vel = state[self.cube_rel_vel_i:self.cube_rel_vel_i+3]
        cartesian_vel = gripper_vel + cube_relative_vel
        return cartesian_vel
    
    def check_gripper_open(self, state):
        thresh = self.gripper_open_thresh
        distance_between_grippers = self.get_distance_grippers(state)
        success = distance_between_grippers > self.gripper_open_thresh
        dist_clipped  = np.clip(distance_between_grippers, 0, thresh)
        reward = np.clip(-(0.1 - dist_clipped), -1, -thresh)
        return success, reward
    
    def check_cube_between_grippers(self, state):
        thresh = self.cube_between_gripper_dist_thresh
        dist_gripper_cube = np.linalg.norm(self.get_gripper_pos(state) - self.get_cube_pos(state))
        success = dist_gripper_cube < thresh
        reward = np.clip(-self.scale_cube_between*dist_gripper_cube, -1, 0)
        return success, reward
    
    def check_cube_between_grippers_easy(self, state):
        # TODO: proper thresholds, check works....
        # check gripper open
        open_success, open_rew = self.check_gripper_open(state)
        # print("gripper_open_success: ", open_success)
        # x dist
        x_dist = np.abs(self.get_gripper_pos(state)[0] - self.get_cube_pos(state)[0])
        # print("x_dist: ", x_dist)
        x_pass = x_dist < 0.03
        # y dist
        y_dist = np.abs(self.get_gripper_pos(state)[1] - self.get_cube_pos(state)[1])
        # print("y_dist: ", y_dist)
        y_pass = y_dist < 0.025
        # z_dist
        z_dist = np.abs(self.get_gripper_pos(state)[2] - self.get_cube_pos(state)[2])
        # print("z_dist: ", z_dist)
        z_pass = z_dist < 0.03
        # overall
        success = open_success and x_pass and y_pass and z_pass
        return success, None
        
    
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
    
    def check_gripper_at_target(self, state):
        gripper_pos = self.get_gripper_pos(state)
        target_pos = self.get_target_pos(state)
        dist = self.distance(gripper_pos, target_pos)
        success = (dist < self.gripper_target_dist_thresh)
        reward = np.clip(-dist, -1.0, 0.0)
        return success, reward
    
    def check_gripper_stationary(self, state):
        gripper_vel = self.get_gripper_vel(state)
        stationary = np.linalg.norm(gripper_vel) < self.gripper_stationary_thresh
        success = stationary
        reward = -1 + stationary
        return success, reward
    
    def check_gripper_open_stationary(self, state):
        open_success, open_rew = self.check_gripper_open(state)
        stationary_success, stationary_rew = self.check_gripper_stationary(state)
        success = open_success and stationary_success
        reward = np.clip(open_rew + stationary_rew, -1, 0)
        return success, reward
    
    def check_cube_moving_to_target(self, state):
        cube_pos = self.get_cube_pos(state)
        target_pos = self.get_target_pos(state)
        direction_to_target = target_pos - cube_pos
        cube_vel = self.get_cube_vel(state)
        # check velocity sufficiently high
        cube_moving_success = np.linalg.norm(cube_vel) > self.cube_moving_vel_threshold
        # check angle
        v1_u = direction_to_target / np.linalg.norm(direction_to_target)
        v2_u = cube_vel / np.linalg.norm(cube_vel)
        angle = np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
        # check velocity success
        direction_success = (angle <= self.velocity_angle_thresh)
        vel_success = direction_success and cube_moving_success
        # check cube at target
        cube_at_target_success, _ = self.check_cube_at_target(state)
        # overall success
        success = vel_success or cube_at_target_success
        reward = np.clip(-angle, -1, 0)
        return success, reward

    
    def distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)

# TODO: add velocity penalty??? Arm should be still when task complete??  


valid_tasks = ['move_cube_to_target',
                'pick_up_cube',
                    'move_gripper_to_cube',
                    "grasp_cube",
                        # "open_gripper",
                        "cube_between_grippers",
                        "close_gripper_cube",
                    "lift_cube",
                "place_cube_at_target",
                    "move_cube_towards_target_grasp",       # "move_gripper_to_target_grasp",
            ]


class Task():
    '''
    WARNING: Task should only exist within a single rollout (i.e. re-initialize each time the nvironmnet is reset!!)
    - Note: very important that final child in sequence has identical success condition to parent (child MUST complete parent)
    '''
    def __init__(self, parent_task=None, subtask_cls_seq=[], level=0, use_dense_reward_lowest_level=False, complete_thresh=3):
        assert self.name in valid_tasks
        self.parent_task = parent_task
        self.level = level
        self.use_dense_reward_lowest_level = use_dense_reward_lowest_level
        self.complete_thresh = complete_thresh         

        self.next_task = None
            
        # state parser
        self.state_parser = FetchPickPlaceStateParser()
        
        # completion tracking - within episode stats!!
        self.complete = False
        self.success_count = 0

    def set_subtask_sequence(self, subtask_sequence):
        # init subtasks
        self.subtask_sequence = subtask_sequence
        # set next tasks
        if len(self.subtask_sequence) > 1:
            for i in range(len(self.subtask_sequence)-1):
                self.subtask_sequence[i].set_next_task(self.subtask_sequence[i+1])
        # set dense rewards
        if len(self.subtask_sequence) == 0 and self.use_dense_reward_lowest_level:
            self.use_dense_reward = True
        else:
            self.use_dense_reward = False
    
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
                    parent_success, _ = task.parent_task.check_success_reward(current_state)
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
        raise NotImplementedError("Subclass must implement abstract method")
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
    
    def record_relations(self):
        '''
        Records any relations and their distance from task
        '''
        # parents
        def set_parent_relations(task, distance=1):
            parents_dict = {}
            parent_task = task.parent_task
            if parent_task is not None:
                parents_dict[parent_task.name] = distance
                parents_dict.update(set_parent_relations(parent_task, distance=distance+1))
            return parents_dict
        parents_dict = set_parent_relations(self, distance=1)
        
        # children
        def set_child_relations(task, distance=1):
            child_dict = {}
            if len(task.subtask_sequence) > 0:
                for child_task in task.subtask_sequence:
                    child_dict[child_task.name] = distance
                    child_dict.update(set_child_relations(child_task, distance=distance+1))
            return child_dict
        child_dict = set_child_relations(self, distance=1)
        # set relations
        self.relations = {'parents': parents_dict, 'children': child_dict}
        
    def record_child_proportions(self):
        '''
        Records what percent of the task each child completes
        '''
        # children
        def set_child_proportions(task, task_prop=1.0):
            assert task_prop <= 1.0
            child_props = {}
            if len(task.subtask_sequence) > 0:
                for child_task in task.subtask_sequence:
                    child_prop = (1.0 / len(task.subtask_sequence)) * task_prop
                    child_props[child_task.name] = child_prop
                    child_props.update(set_child_proportions(child_task, task_prop=child_prop))
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


'''
FetchPickAndPlace
 
'''
 
class MoveCubeToTargetTask(Task):
    '''
    The parent task for the FetchPickPlace environment
    '''
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "move_cube_to_target"
        self.str_description = "Move cube to target"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_at_target(current_state)
        reward = self.binary_reward(success)
        return success, reward

    
class PickUpCubeTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "pick_up_cube"
        self.str_description = "Pick up cube"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
        
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
        self.str_description = "Go to cube"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_gripper_above_cube(current_state)
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
        cube_pos = self.state_parser.get_cube_pos(state) + self.state_parser.cube_height_offset
        gripper_pos = self.state_parser.get_gripper_pos(state)
        direction = cube_pos - gripper_pos
        gripper_open = True
        return direction, gripper_open

class GraspCubeTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "grasp_cube"
        self.str_description = "Grasp cube"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_grasped(current_state)
        if self.use_dense_reward:
            reward = dense_reward
        else:
            reward = self.binary_reward(success)
        return success, reward

 
class LiftCubeTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "lift_cube"
        self.str_description = "Lift cube"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_cube_lifted_and_grasped(current_state)
        # reward
        if self.use_dense_reward:
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
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
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
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "move_cube_towards_target_grasp"
        self.str_description = "Move cube to target while grasping"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
    
    def check_success_reward(self, current_state):
        if self.use_dense_reward:
            success, dense_reward = self.state_parser.check_cube_at_target(current_state)
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
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "open_gripper"
        self.str_description = "Open gripper"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_gripper_open_stationary(current_state)
        # reward
        if self.use_dense_reward:
            reward = dense_reward
        else:
            reward = self.binary_reward(success)
        return success, reward
    
    def get_oracle_action(self, state):
        return np.array([0, 0, 0]), True
    

class CubeBetweenGripperTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "cube_between_grippers"
        self.str_description = "Put grippers around cube"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_cube_between_grippers_easy(current_state)
        # reward
        if self.use_dense_reward:
            reward = dense_reward
        else:
            reward = self.binary_reward(success)
        return success, reward
    
    def get_oracle_action(self, state):
        cube_between_gripper, _ = self.state_parser.check_cube_between_grippers_easy(state)
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
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "close_gripper_cube"
        self.str_description = "Close gripper around cube"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
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




'''
FetchReach - Need a major revamp to get this to work!!!!
'''

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
    