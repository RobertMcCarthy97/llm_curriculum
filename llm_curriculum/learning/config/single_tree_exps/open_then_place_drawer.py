from llm_curriculum.learning.config.single_tree_exps.base_config import (
    get_config as get_base_config,
)


def get_config():
    """Returns the default config"""
    config = get_base_config()

    config.is_closed_on_reset = True
    config.cube_pos_on_reset = "table"

    config.max_ep_len = 80
    config.total_timesteps = 1e6

    config.dense_rew_tasks = [
        "move_gripper_to_drawer",
        "move_gripper_to_cube",
        "move_cube_over_drawer",
    ]
    config.high_level_task_names = ["open_then_place_in_drawer"]

    config.do_seperate_policy_eval = False  # sometimes doesn't reach...

    config.wandb.name = "open_then_place_in_drawer-single_tree"

    return config
