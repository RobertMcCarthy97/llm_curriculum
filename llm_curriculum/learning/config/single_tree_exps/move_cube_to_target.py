from llm_curriculum.learning.config.single_tree_exps.base_config import (
    get_config as get_base_config,
)


def get_config():
    """Returns the default config"""
    config = get_base_config()

    config.drawer_env = False
    config.is_closed_on_reset = True
    config.cube_pos_on_reset = "table"

    config.dense_rew_tasks = ["move_gripper_to_cube", "move_cube_towards_target_grasp"]
    config.high_level_task_names = ["move_cube_to_target"]

    config.wandb.name = "move_cube_to_target-single_tree"

    return config
