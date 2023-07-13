from llm_curriculum.learning.config.low_level_only.base_config import (
    get_config as get_base_config,
)


def get_config():
    """Returns the default config"""
    config = get_base_config()

    config.drawer_env = True
    config.is_closed_on_reset = False
    config.cube_pos_on_reset = "table"

    config.dense_rew_tasks = ["move_gripper_to_cube", "move_cube_over_drawer"]
    config.high_level_task_names = ["place_cube_open_drawer"]

    config.wandb.name = "place_cube_in_open_drawer-low_level_only"

    return config
