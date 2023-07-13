# information that does not change during the episode
static_obs = {
    "parallel_gripper": {
        "max_distance_between": 0.2,
    },
    "cube": {
        "width": 0.1,
    },
    "drawer": {
        "max_open_distance": 1.0,
    },
    "table": {
        "top_surface_height": 0.5,
    },
}

# information that could change during the episode
dynamic_obs = {
    "parallel_gripper": {
        "center_coordinate": [0.0, -1.0, 1.0],
        "distance_between": 0.5,  # the distance between the two fingers
        "objects_between": ["cube"],  # One of ["cube", "drawer_handle", None]
    },
    "cube": {
        "center_coordinate": [0.0, -1.0, 1.0],
        "semantic_position": "in_drawer",  # One of ["in_drawer", "on_top_of_drawer", "on_table", "grasped", None]
    },
    "drawer": {
        "open_distance": 0.0,  # 0.0 means closed, 1.0 means fully open
        "handle_center_coordinate": [2.0, 2.0, 1.5],
    },
}
