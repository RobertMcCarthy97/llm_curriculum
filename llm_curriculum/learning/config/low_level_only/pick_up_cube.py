from llm_curriculum.learning.config.low_level_only.base_config import (
    get_config as get_base_config,
)


def get_config():
    """Returns the default config"""
    config = get_base_config()

    config.drawer_env = False
    config.is_closed_on_reset = True
    config.cube_pos_on_reset = "table"

    config.dense_rew_tasks = ["move_gripper_to_cube"]
    config.high_level_task_names = ["pick_up_low"]

    config.wandb.name = "pick_up_cube-low_level_only"

    return config
