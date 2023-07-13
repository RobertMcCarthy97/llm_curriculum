from llm_curriculum.learning.config.zero_shot_pretrain.base_config import (
    get_config as get_base_config,
)


def get_config():
    """Returns the default config"""
    config = get_base_config()

    config.is_closed_on_reset = False
    config.cube_pos_on_reset = "in_drawer"

    config.dense_rew_tasks = ["move_gripper_to_drawer"]
    config.high_level_task_names = ["close_drawer"]

    config.wandb.name = "close_drawer-pretrained"

    return config
