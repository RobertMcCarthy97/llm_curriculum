from llm_curriculum.learning.config.low_level_only.base_config import (
    get_config as get_base_config,
)


def get_config():
    """Returns the default config"""
    config = get_base_config()

    config.drawer_env = True
    config.is_closed_on_reset = True
    config.cube_pos_on_reset = "table"
    config.max_ep_len = 80

    config.dense_rew_tasks = [
        "move_gripper_to_drawer",
        "move_gripper_to_cube",
        "move_cube_over_drawer",
    ]
    config.high_level_task_names = ["open_then_place_drawer_low"]

    config.wandb.name = "open_then_place_drawer-low_level_only"

    return config
