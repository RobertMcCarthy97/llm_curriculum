from llm_curriculum.learning.config.single_tree_exps.base_config import (
    get_config as get_base_config,
)


def get_config():
    """Returns the default config"""
    config = get_base_config()

    config.is_closed_on_reset = False
    config.cube_pos_on_reset = "on_drawer"

    config.dense_rew_tasks = ["move_gripper_to_cube", "move_cube_over_drawer"]
    config.high_level_task_names = ["cube_on_drawer_to_cube_in_drawer"]

    config.total_timesteps = 3e5
    config.wandb.name = "v1-on_drawer_to_in_drawer-pretrain-v1"

    return config
