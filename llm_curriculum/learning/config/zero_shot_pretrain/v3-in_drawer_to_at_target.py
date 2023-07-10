from llm_curriculum.learning.config.single_tree_exps.base_config import (
    get_config as get_base_config,
)


def get_config():
    """Returns the default config"""
    config = get_base_config()

    config.is_closed_on_reset = False
    config.cube_pos_on_reset = "in_drawer"

    config.dense_rew_tasks = [
        "move_gripper_to_cube_in_drawer",
        "move_cube_towards_target_grasp",
    ]
    config.high_level_task_names = ["cube_in_drawer_to_cube_at_target"]

    config.do_seperate_policy_eval = (
        False  # Oracle actions not working when cube starts in drawer
    )

    config.total_timesteps = 5e5
    config.wandb.name = "v3-in_drawer_to_at_target-pretrain-v3"

    return config
