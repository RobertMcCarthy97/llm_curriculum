import numpy as np

#######################
# State parser
#######################

class RectangularVolume:
    '''
        You can define a rectangular volume using two points:
        the first point (x1, y1, z1) represents the minimum values (lower left corner) and the second point (x2, y2, z2) represents the maximum values (upper right corner).
        This assumes that the rectangular volume is aligned with the axis. 
        '''
    def __init__(self, min_point, max_point):
        self.min_point = min_point
        self.max_point = max_point

    def contains(self, point):
        return (self.min_point[0] <= point[0] <= self.max_point[0] and
                self.min_point[1] <= point[1] <= self.max_point[1] and
                self.min_point[2] <= point[2] <= self.max_point[2])
    
    def get_center(self):
        return (self.min_point + self.max_point) / 2
    

class TempDemoDrawerStateParser():
    '''
    Takes in state observations and returns reward and success for different conditions.

    # TODO: pass in obs space during init also...
    '''
    def __init__(self):
        ## position of info in observation vector
        self.drawer_open_i = 0

        ## additional env info
        self.drawer_static_pos = np.array([None, None, None]) # TODO: Daniel
        # drawer volume
        '''
        "for this setup, when the cube is inside the drawer and the drawer is closed, the range is (1.21, 1.38), (0.83, 0.96) and (0.48, 0.55) respectively." 
        '''
        self.drawer_min_point = np.array([1.21, 0.98, 0.48])
        self.drawer_max_point = np.array([1.38, 1.11, 0.55])
        self.drawer_static_rect_volume = RectangularVolume(self.drawer_min_point, self.drawer_max_point)
        # which of x, y, z does drawer open along?
        self.open_along_dim = 1
        self.handle_length_dim = 0
        self.over_drawer_height_offset = None # TODO: Daniel
        self.handle_offset_from_drawer_pos = np.array([0, -0.03, 0]) # TODO: this is incorrect
        self.handle_length = 0.05
        self.handle_offset_vol = None # TODO: Daniel
        self.above_handle_height_offset = None

        ## task related thresholds
        self.drawer_open_thresh = -0.1
        self.drawer_closed_thresh = -0.04
        self.gripper_handle_height_thresh = None

    def get_drawer_static_center(self, state):
        return self.drawer_static_rect_volume.get_center().copy()
    
    def get_drawer_static_min_point(self, state):
        return self.drawer_min_point.copy()
    
    def get_drawer_static_max_point(self, state):
        return self.drawer_max_point.copy()
    
    def get_drawer_dynamic_center(self, state):
        static_center = self.get_drawer_static_center(state)
        open_y_pos = self.get_drawer_open_magnitude(state)
        dynamic_center = static_center.copy()
        dynamic_center[self.open_along_dim] += open_y_pos
        return dynamic_center.copy()
    
    def get_drawer_open_magnitude(self, state):
        return state[self.drawer_open_i].copy()
    
    def get_drawer_dynamic_min_point(self, state):
        open_mag = self.get_drawer_open_magnitude(state)
        dynamic_min = self.get_drawer_static_min_point(state)
        dynamic_min[self.open_along_dim] += open_mag
        return dynamic_min.copy()
    
    def get_drawer_dynamic_max_point(self, state):
        open_mag = self.get_drawer_open_magnitude(state)
        dynamic_max = self.get_drawer_static_max_point(state)
        dynamic_max[self.open_along_dim] += open_mag
        return dynamic_max.copy()
    
    def get_drawer_dynamic_rect_volume(self, state):
        dynamic_min_point = self.get_drawer_dynamic_min_point(state)
        dynamic_max_point = self.get_drawer_dynamic_max_point(state)
        dynamic_rect_volume = RectangularVolume(dynamic_min_point, dynamic_max_point)
        return dynamic_rect_volume
    
    def get_over_drawer_dynamic_rect_volume(self, state):
        '''
        Note, currently includes volume in drawer aswell as above...
        '''
        dynamic_min_point = self.get_drawer_dynamic_min_point(state)
        dynamic_max_point = self.get_drawer_dynamic_max_point(state)
        dynamic_max_point[2] += self.over_drawer_height_offset
        dynamic_above_rect_volume = RectangularVolume(dynamic_min_point, dynamic_max_point) # TODO: should be able to just copy and add offset??
        return dynamic_above_rect_volume
    
    def get_handle_pos(self, state):
        drawer_dynamic_pos = self.get_drawer_dynamic_center(state)
        handle_pos = drawer_dynamic_pos + self.handle_offset_from_drawer_pos
        return handle_pos
    
    def get_handle_length(self, state):
        return self.handle_length.copy()
    
    def get_handle_min_max_point(self, state):
        handle_length = self.get_handle_length(state)
        handle_pos = self.get_handle_pos(state)
        # min
        handle_min_point = handle_pos.copy()
        handle_min_point[self.handle_length_dim] -= handle_length / 2
        mask = np.arange(3) != self.handle_length_dim
        handle_min_point[mask] -= self.handle_vol_offset
        # max
        handle_max_point = handle_pos.copy()
        handle_max_point[self.handle_length_dim] += handle_length / 2
        mask = np.arange(3) != self.handle_length_dim
        handle_max_point[mask] += self.handle_vol_offset
        raise NotImplementedError
        return handle_min_point, handle_max_point

    def get_handle_rect_volume(self, state):
        handle_min_point, handle_max_point = self.get_handle_min_max_point(state)
        handle_rect_volume = RectangularVolume(handle_min_point, handle_max_point)
        return handle_rect_volume
    
    def get_above_handle_rect_volume(self, state):
        handle_min_point, handle_max_point = self.get_handle_min_max_point(state)
        above_handle_min_point = handle_min_point.copy()
        above_handle_max_point = handle_max_point.copy()
        above_handle_min_point[2] += self.above_handle_height_offset
        above_handle_max_point[2] += self.above_handle_height_offset
        above_handle_rect_volume = RectangularVolume(above_handle_min_point, above_handle_max_point)
        raise NotImplementedError
        return above_handle_rect_volume
    
    def check_drawer_open(self, state):
        open_pos = self.get_drawer_open_magnitude(state)
        success = open_pos < self.drawer_open_thresh
        reward = np.clip(open_pos, -1, 0) # TODO
        raise NotImplementedError
        return success, reward
    
    def check_drawer_closed(self, state):
        open_pos = self.get_drawer_open_magnitude(state)
        success = open_pos > self.drawer_closed_thresh
        reward = np.clip(open_pos, -1, 0) # TODO
        raise NotImplementedError
        return success, reward

    def check_cube_in_drawer(self, state):
        cube_pos = self.get_cube_pos(state)
        drawer_dynamic_rect_vol = self.get_drawer_dynamic_rect_volume(state)
        success = drawer_dynamic_rect_vol.contains(cube_pos)
        reward = self.binary_reward(success)
        raise NotImplementedError
        return success, reward
    
    def check_cube_over_dynamic_drawer(self, state):
        '''
        For now, just check if cube is over drawer, regardless of whether open or closed
        - In future, check only if over open portion?
        '''
        cube_pos = self.get_cube_pos(state)
        dynamic_above_rect_volume = self.get_over_drawer_dynamic_rect_volume(state)
        success = dynamic_above_rect_volume.contains(cube_pos)
        reward = self.binary_reward(success)
        raise NotImplementedError, "Implement dense reward"
        return success, reward
    
    def get_static_above_rect_volume(self, state):
        static_min = self.get_drawer_static_min_point(state)
        static_max = self.get_drawer_static_max_point(state)
        static_max[2] += self.over_drawer_height_offset
        static_above_rect_volume = RectangularVolume(static_min, static_max)
        return static_above_rect_volume
    
    def check_cube_over_drawer_top(self, state):
        cube_pos = self.get_cube_pos(state)
        static_above_rect_volume = self.get_static_above_rect_volume(state)
        success = static_above_rect_volume.contains(cube_pos)
        reward = self.binary_reward(success)
        raise NotImplementedError, "Implement dense reward"
        return success, reward
        
    
    def check_gripper_above_handle(self, state):
        gripper_pos = self.get_gripper_pos(state)
        above_handle_rect_vol = self.get_above_handle_rect_volume(state)
        success = above_handle_rect_vol.contains(gripper_pos)
        reward = self.binary_reward(success)
        raise NotImplementedError
        return success, reward

    def check_handle_grasped(self, state):
        '''
        check if handle axis is inside gripper
        '''
        # axis
        gripper_left_x_pos, gripper_right_x_pos = self.get_gripper_left_right_x_pos(state)
        handle_axis_x_pos = self.get_handle_axis_x_pos(state)
        axis_success = (gripper_left_x_pos < handle_axis_x_pos) and (handle_axis_x_pos < gripper_right_x_pos)
        # height
        gripper_height = self.get_gripper_pos(state)[2]
        handle_height = self.get_handle_pos(state)[2]
        height_success = (handle_height < gripper_height + self.gripper_handle_height_thresh) and (handle_height > gripper_height - self.gripper_handle_height_thresh)
        # overall
        success = axis_success and height_success
        reward = self.binary_reward(success)
        raise NotImplementedError
        return success, reward


#######################
# Task classes
#######################

################## Place cube in drawer

class PlaceCubeDrawerTask(Task):
    '''
    The parent task for the FetchPickPlace environment
    '''
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "place_cube_drawer"
        self.str_description = "Place cube in drawer"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_in_drawer(current_state)
        reward = self.binary_reward(success)
        return success, reward
    
class MoveCubeOverDrawerTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "move_cube_over_drawer"
        self.str_description = "Move cube over drawer"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)

    def check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_cube_over_drawer(current_state)
        if self.use_dense_reward_lowest_level:
            reward = dense_reward
        else:
            reward = self.binary_reward(success)
            assert False, "use dense as sparse too difficult"
        return success, reward
    
    def get_oracle_action(self, state):
        '''
        Assume grasping cube and drawer open
        '''
        # move gripper above handle
        drawer_pos = self.state_parser.get_drawer_dynamic_center(state) + self.state_parser.drawer_height_offset
        gripper_pos = self.state_parser.get_gripper_pos(state)
        direction = drawer_pos - gripper_pos
        gripper_open = False
        return direction, gripper_open

    
class ReleaseCubeInDrawerTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "release_cube_in_drawer"
        self.str_description = "Release cube in drawer"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)

    def check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_in_drawer(current_state)
        reward = self.binary_reward(success)
        return success, reward
    
    def get_oracle_action(self, state):
        '''
        Assume grasping cube over open drawer
        '''
        gripper_open = True
        return np.array([0, 0, 0]), gripper_open

################### Open/clsoe drawer

class OpenDrawerTask(Task):
    '''
    The parent task for the FetchPickPlace environment
    '''
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "open_drawer"
        self.str_description = "Open drawer"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, _ = self.state_parser.check_drawer_open(current_state)
        reward = self.binary_reward(success)
        return success, reward
    
class CloseDrawerTask(Task):
    '''
    The parent task for the FetchPickPlace environment
    '''
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "close_drawer"
        self.str_description = "Close drawer"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, _ = self.state_parser.check_drawer_open(current_state)
        reward = self.binary_reward(success)
        return success, reward

    
class MoveGripperToDrawerTask(Task):
    '''
    TODO: refactor as MoveGripperToObjectTask - takes the object as input (so can deal with different objects...)
    '''
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "move_gripper_to_drawer"
        self.str_description = "Go to drawer"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_gripper_above_handle(current_state)
        if self.use_dense_reward:
            if success:
                reward = 0
            else:
                reward = np.clip(dense_reward * 6, -1, 0)
        else:
            reward = self.binary_reward(success)
        return success, reward
    
    def get_oracle_action(self, state):
        # move gripper above handle
        handle_pos = self.state_parser.get_handle_pos(state) + self.state_parser.handle_height_offset
        gripper_pos = self.state_parser.get_gripper_pos(state)
        direction = handle_pos - gripper_pos
        gripper_open = True
        return direction, gripper_open
    
class GraspHandleTask(Task):
    # TODO: implement between and close?
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "grasp_handle"
        self.str_description = "Grasp handle"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_handle_grasped(current_state)
        if self.use_dense_reward:
            reward = dense_reward
        else:
            reward = self.binary_reward(success)
        return success, reward
    
    # TODO: implement oracle action


class PullHandleToOpenTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "pull_handle_to_open"
        self.str_description = "Pull handle to open drawer"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, _ = self.state_parser.check_drawer_open(current_state)
        reward = self.binary_reward(success)
        return success, reward
    
    def get_oracle_action(self, state):
        '''
        Assumes handle already grasped - move gripper in direction drawer opens
        '''
        return np.array([1, 0, 0]), False
    

class PushHandleToCloseTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "push_handle_to_close"
        self.str_description = "Push handle to close drawer"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)

    def check_success_reward(self, current_state):
        success, _ = self.state_parser.check_drawer_closed(current_state)
        reward = self.binary_reward(success)
        return success, reward
    
    def get_oracle_action(self, state):
        '''
        Assumes handle already grasped - move gripper in direction drawer closes
        '''
        return np.array([-1, 0, 0]), False
    

################### Place cube ON drawer

class PlaceCubeOnDrawerTopTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "place_cube_on_drawer_top"
        self.str_description = "Place cube on drawer top"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)
        
    def check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_on_drawer(current_state)
        reward = self.binary_reward(success)
        return success, reward

class MoveCubeOverDrawerTopTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "move_cube_over_drawer_top"
        self.str_description = "Move cube over drawer top"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)

    def check_success_reward(self, current_state):
        success, dense_reward = self.state_parser.check_cube_over_drawer_top(current_state)
        if self.use_dense_reward:
            reward = dense_reward
        else:
            reward = self.binary_reward(success)
            assert False, "use dense as sparse too difficult"
        return success, reward
    
    def get_oracle_action(self, state):
        '''
        Assumes cube grasped
        '''
        gripper_pos = self.state_parser.get_gripper_pos(state)
        above_pos = self.state_parser.get_static_above_rect_volume(state).get_center()
        direction = above_pos - gripper_pos
        gripper_open = False
        return direction, gripper_open
    

class ReleaseCubeOnDrawerTopTask(Task):
    def __init__(self, parent_task=None, level=0, use_dense_reward_lowest_level=False):
        self.name = "release_cube_on_drawer_top"
        self.str_description = "Release cube on drawer top"
        
        super().__init__(parent_task, level, use_dense_reward_lowest_level)

    def check_success_reward(self, current_state):
        success, _ = self.state_parser.check_cube_on_drawer(current_state)
        reward = self.binary_reward(success)
        return success, reward
    
    def get_oracle_action(self, state):
        '''
        Assumes cube already grasped over drawer
        '''
        gripper_open = True
        return np.array([0, 0, 0]), gripper_open