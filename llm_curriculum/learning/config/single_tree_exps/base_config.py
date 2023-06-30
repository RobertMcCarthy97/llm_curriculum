from ml_collections import config_dict
from datetime import datetime
from llm_curriculum.envs.curriculum_manager import (
    SeperateEpisodesCM,
    DummySeperateEpisodesCM,
)
from stable_baselines3 import TD3
from stable_baselines3.common.buffers_custom import SeparatePoliciesReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise

"""
TODO:
- Have different base configs and be able to select between them
"""


def get_config():
    """Returns the default config"""
    # TODO: Figure out how to convert string to class
    # E.g. config.algo = "TD3" -> config.algo = TD3
    config = config_dict.ConfigDict()

    # general
    config.help = False
    config.seed = 0
    # env
    config.drawer_env = True
    config.incremental_reward = False
    config.manual_decompose_p = None
    config.dense_rew_lowest = False
    config.dense_rew_tasks = ["move_gripper_to_cube"]
    config.use_language_goals = False
    config.render_mode = "rgb_array"
    config.use_oracle_at_warmup = False
    config.max_ep_len = 50
    config.use_baseline_env = False
    config.is_closed_on_reset = True
    config.is_cube_inside_drawer_on_reset = False
    # task
    config.single_task_names = []
    config.high_level_task_names = ["pick_up_cube"]
    config.curriculum_manager_cls = SeperateEpisodesCM  # DummySeperateEpisodesCM, SeperateEpisodesCM (CM decides 'decompose_p' based on success rates)
    config.sequenced_episodes = True
    config.contained_sequence = False
    config.initial_state_curriculum_p = 0.0
    # algo
    config.algo = TD3
    config.policy_type = "MlpPolicy"
    config.learning_starts = 1e3
    config.replay_buffer_class = SeparatePoliciesReplayBuffer
    config.replay_buffer_kwargs = {"child_p": 0.2}
    config.total_timesteps = 1e6
    config.device = "cpu"
    config.policy_kwargs = None  # None, {'goal_based_custom_args': {'use_siren': True, 'use_sigmoid': True}}
    config.action_noise = NormalActionNoise
    # TODO: increase batch size??
    # logging
    config.wandb = config_dict.ConfigDict()
    config.wandb.track = True
    config.wandb.project = "llm-curriculum"
    config.wandb.entity = "robertmccarthy11"
    config.wandb.group = "single-tree-exp"
    config.wandb.job_type = "training"
    config.wandb.name = "pickup_cube-default_exp"

    config.log_path = "./logs/" + f"{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}"
    config.save_models = True
    config.eval_policy = True
    config.eval_traversal_modes = ["train", "leaf", "exploit"]

    # config.exp_group = "merge-validation"
    config.info_keywords = ("is_success", "overall_task_success", "active_task_level")

    # assert False, "increase episode length!"

    return config
