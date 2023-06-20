import numpy as np


class RectangularVolume:
    """
    You can define a rectangular volume using two points:
    the first point (x1, y1, z1) represents the minimum values (lower left corner) and the second point (x2, y2, z2) represents the maximum values (upper right corner).
    This assumes that the rectangular volume is aligned with the axis.
    """

    def __init__(self, min_point, max_point):
        self.min_point = min_point
        self.max_point = max_point

    def contains(self, point):
        return (
            self.min_point[0] <= point[0] <= self.max_point[0]
            and self.min_point[1] <= point[1] <= self.max_point[1]
            and self.min_point[2] <= point[2] <= self.max_point[2]
        )

    def get_center(self):
        return (self.min_point + self.max_point) / 2


################################
# Core state parser
################################


class CoreStateParser:
    """
    Takes in state observations and returns reward and success for different conditions.

    # TODO: pass in obs space during init also...
    """

    def __init__(self):
        self.expected_obs_size = 28

        # position of info in observation vector
        self.gripper_pos_i = 0
        self.cube_pos_i = 3
        self.right_gripper_i = 9
        self.left_gripper_i = 10
        self.target_i = 25  # TODO: this will change if add drawer...
        self.gripper_vel_i = 20
        self.cube_rel_vel_i = 14  # velocity relative to gripper...

        # additional env info
        self.cube_width = 0.025
        self.table_height = 0.425

        # task related thresholds
        self.cube_height_offset = np.array([0, 0, 0.05])  # 0.05
        self.gripper_closed_cube_thresh = (
            0.05  # distance between grippers when grasping cube
        )
        self.cube_lifted_thresh = self.table_height + 2.5 * self.cube_width
        self.cube_between_gripper_dist_thresh = self.cube_width
        self.gripper_open_thresh = 1.8 * self.cube_width
        self.gripper_stationary_thresh = 0.001
        self.cube_moving_vel_threshold = 0.005

        self.cube_target_dist_threshold = 0.05
        self.gripper_target_dist_threshold = 0.05
        self.gripper_cube_dist_thresh = 0.05

        self.scale_cube_between = 10
        self.scale_cube_lifted = 10

        self.velocity_angle_thresh = 45  # degrees

    def check_obs_space(self, obs_space):
        assert (
            obs_space.shape[0] == self.expected_obs_size
        ), f"obs size should be {self.expected_obs_size}, not {obs_space.shape[0]}"

    def get_gripper_pos(self, state):
        return state[self.gripper_pos_i : self.gripper_pos_i + 3]

    def get_cube_pos(self, state):
        if state.shape[0] < 28:
            assert False
        return state[self.cube_pos_i : self.cube_pos_i + 3]

    def get_target_pos(self, state):
        return state[self.target_i : self.target_i + 3]

    def get_distance_grippers(self, state):
        return state[self.right_gripper_i] + state[self.left_gripper_i]

    def get_gripper_vel(self, state):
        cartesian_vel = state[self.gripper_vel_i : self.gripper_vel_i + 3]
        return cartesian_vel

    def get_cube_vel(self, state):
        gripper_vel = self.get_gripper_vel(state)
        cube_relative_vel = state[self.cube_rel_vel_i : self.cube_rel_vel_i + 3]
        cartesian_vel = gripper_vel + cube_relative_vel
        return cartesian_vel

    def check_gripper_open(self, state):
        thresh = self.gripper_open_thresh
        distance_between_grippers = self.get_distance_grippers(state)
        success = distance_between_grippers > self.gripper_open_thresh
        dist_clipped = np.clip(distance_between_grippers, 0, thresh)
        reward = np.clip(-(0.1 - dist_clipped), -1, -thresh)
        return success, reward

    def check_cube_between_grippers(self, state):
        thresh = self.cube_between_gripper_dist_thresh
        dist_gripper_cube = np.linalg.norm(
            self.get_gripper_pos(state) - self.get_cube_pos(state)
        )
        success = dist_gripper_cube < thresh
        reward = np.clip(-self.scale_cube_between * dist_gripper_cube, -1, 0)
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
        reward = np.clip(-self.scale_cube_lifted * (thresh - cube_z), -1, 0)
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
        (
            grip_closed_cube_success,
            grip_closed_cube_rew,
        ) = self.check_grippers_closed_cube_width(state)
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
        success = dist < self.gripper_cube_dist_thresh
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
        success = dist < self.gripper_target_dist_thresh
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
        direction_success = angle <= self.velocity_angle_thresh
        vel_success = direction_success and cube_moving_success
        # check cube at target
        cube_at_target_success, _ = self.check_cube_at_target(state)
        # overall success
        success = vel_success or cube_at_target_success
        reward = np.clip(-angle, -1, 0)
        return success, reward

    def distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)


#############################
# Drawer State-parser
#############################


class DrawerStateParser(CoreStateParser):
    """
    Takes in state observations and returns reward and success for different conditions.

    # TODO: pass in obs space during init also...
    """

    def __init__(self):
        super().__init__()
        self.drawer_obs_size = 2  # TODO: make this a param

        ## position of info in observation vector
        self.drawer_open_i = self.expected_obs_size
        # Adjust for drawer
        self.expected_obs_size += self.drawer_obs_size

        ## additional env info
        self.drawer_static_pos = np.array([None, None, None])  # TODO: Daniel

        # drawer volume
        """
        "for this setup, when the cube is inside the drawer and the drawer is closed, the range is (1.21, 1.38), (0.83, 0.96) and (0.48, 0.55) respectively." 
        """
        self.drawer_min_point = np.array([1.21, 0.98, 0.48])
        self.drawer_max_point = np.array([1.38, 1.11, 0.55])
        self.drawer_static_rect_volume = RectangularVolume(
            self.drawer_min_point, self.drawer_max_point
        )

        self.open_along_dim = 1
        self.handle_length_dim = 0

        # handle
        self.over_drawer_height_offset = 0.1
        self.handle_offset_from_drawer_front = np.array([0, -0.03, 0])
        self.handle_length = 0.05
        self.handle_vol_buffer_offset = 0.01  # TODO: Daniel
        self.above_handle_height_offset = np.array([0, 0, 0.05])

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
        """
        Note, currently includes volume in drawer aswell as above...
        """
        dynamic_min_point = self.get_drawer_dynamic_min_point(state)
        dynamic_max_point = self.get_drawer_dynamic_max_point(state)
        dynamic_max_point[2] += self.over_drawer_height_offset
        dynamic_above_rect_volume = RectangularVolume(
            dynamic_min_point, dynamic_max_point
        )  # TODO: should be able to just copy and add offset??
        return dynamic_above_rect_volume

    def get_drawer_front_center(self, state):
        # drawer opens along y axis
        x = self.get_drawer_static_center(state)[0]
        y = self.get_drawer_dynamic_min_point(state)[1]
        z = self.get_drawer_static_center(state)[2]
        return np.array([x, y, z]).copy()

    def get_handle_pos(self, state):
        import pdb

        pdb.set_trace()
        handle_pos = (
            self.get_drawer_front_center(state) + self.handle_offset_from_drawer_front
        )
        return handle_pos.copy()

    def get_handle_length(self, state):
        return self.handle_length

    def get_handle_min_max_point(self, state):
        handle_length = self.get_handle_length(state)
        handle_pos = self.get_handle_pos(state)
        # min
        handle_min_point = handle_pos.copy()
        handle_min_point[self.handle_length_dim] -= handle_length / 2
        mask = np.arange(3) != self.handle_length_dim
        handle_min_point[mask] -= self.handle_vol_buffer_offset
        # max
        handle_max_point = handle_pos.copy()
        handle_max_point[self.handle_length_dim] += handle_length / 2
        mask = np.arange(3) != self.handle_length_dim
        handle_max_point[mask] += self.handle_vol_buffer_offset
        return handle_min_point, handle_max_point

    def get_handle_rect_volume(self, state):
        handle_min_point, handle_max_point = self.get_handle_min_max_point(state)
        handle_rect_volume = RectangularVolume(handle_min_point, handle_max_point)
        return handle_rect_volume

    def get_above_handle_rect_volume(self, state):
        handle_min_point, handle_max_point = self.get_handle_min_max_point(state)
        above_handle_min_point = (
            handle_min_point.copy() + self.above_handle_height_offset
        )
        above_handle_max_point = (
            handle_max_point.copy() + self.above_handle_height_offset
        )
        above_handle_rect_volume = RectangularVolume(
            above_handle_min_point, above_handle_max_point
        )
        return above_handle_rect_volume

    def check_drawer_open(self, state):
        open_pos = self.get_drawer_open_magnitude(state)
        success = open_pos < self.drawer_open_thresh
        reward = None  # TODO
        return success, reward

    def check_drawer_closed(self, state):
        open_pos = self.get_drawer_open_magnitude(state)
        success = open_pos > self.drawer_closed_thresh
        reward = np.clip(open_pos, -1, 0)  # TODO
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
        """
        For now, just check if cube is over drawer, regardless of whether open or closed
        - In future, check only if over open portion?
        """
        cube_pos = self.get_cube_pos(state)
        dynamic_above_rect_volume = self.get_over_drawer_dynamic_rect_volume(state)
        success = dynamic_above_rect_volume.contains(cube_pos)
        reward = self.binary_reward(success)
        raise NotImplementedError("Implement dense reward")
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
        raise NotImplementedError("Implement dense reward")
        return success, reward

    def check_gripper_above_handle(self, state):
        gripper_pos = self.get_gripper_pos(state)
        above_handle_rect_vol = self.get_above_handle_rect_volume(state)
        success = above_handle_rect_vol.contains(gripper_pos)
        reward = None  # TODO
        return success, reward

    def check_handle_grasped(self, state):
        """
        check if handle axis is inside gripper
        """
        # axis
        gripper_left_x_pos, gripper_right_x_pos = self.get_gripper_left_right_x_pos(
            state
        )
        handle_axis_x_pos = self.get_handle_axis_x_pos(state)
        axis_success = (gripper_left_x_pos < handle_axis_x_pos) and (
            handle_axis_x_pos < gripper_right_x_pos
        )
        # height
        gripper_height = self.get_gripper_pos(state)[2]
        handle_height = self.get_handle_pos(state)[2]
        height_success = (
            handle_height < gripper_height + self.gripper_handle_height_thresh
        ) and (handle_height > gripper_height - self.gripper_handle_height_thresh)
        # overall
        success = axis_success and height_success
        reward = self.binary_reward(success)
        raise NotImplementedError
        return success, reward
