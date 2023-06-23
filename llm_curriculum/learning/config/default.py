from ml_collections import config_dict
from datetime import datetime
from llm_curriculum.envs.curriculum_manager import SeperateEpisodesCM
from stable_baselines3 import TD3
from stable_baselines3.common.buffers_custom import SeparatePoliciesReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise


def get_env_config():
    """Returns the default config"""
    # TODO: Refactor get_config to use this function
    config = config_dict.ConfigDict()

    # env
    config.manual_decompose_p = None
    config.dense_rew_lowest = False
    config.dense_rew_tasks = ["move_gripper_to_cube"]
    config.use_language_goals = False
    config.render_mode = "rgb_array"
    config.use_oracle_at_warmup = False
    config.max_ep_len = 50
    config.use_baseline_env = False
    # task
    config.single_task_names = []
    config.high_level_task_names = ["move_cube_to_target"]
    config.curriculum_manager_cls = SeperateEpisodesCM
    config.sequenced_episodes = True
    config.contained_sequence = False

    return config


def get_config():
    """Returns the default config"""
    # TODO: Figure out how to convert string to class
    # E.g. config.algo = "TD3" -> config.algo = TD3
    config = config_dict.ConfigDict()

    # general
    config.help = False
    config.seed = 0
    # env
    config.drawer_env = False
    config.incremental_reward = False
    config.manual_decompose_p = None
    config.dense_rew_lowest = False
    config.dense_rew_tasks = ["move_gripper_to_cube"]
    config.use_language_goals = False
    config.render_mode = "rgb_array"
    config.use_oracle_at_warmup = False
    config.max_ep_len = 50
    config.use_baseline_env = False
    # task
    config.single_task_names = []
    config.high_level_task_names = ["move_cube_to_target"]
    config.curriculum_manager_cls = SeperateEpisodesCM
    config.sequenced_episodes = True
    config.contained_sequence = False
    # algo
    config.algo = TD3
    config.policy_type = "MlpPolicy"
    config.learning_starts = 1e3
    config.replay_buffer_class = SeparatePoliciesReplayBuffer
    config.replay_buffer_kwargs = {"child_p": 0.2}
    config.total_timesteps = 1e6
    config.device = "cpu"
    config.policy_kwargs = None
    config.action_noise = NormalActionNoise
    # logging
    config.wandb = config_dict.ConfigDict()
    config.wandb.track = False
    config.wandb.project = "llm_curriculum"
    config.wandb.entity = "ucl-air-lab"
    config.wandb.group = "default"
    config.wandb.job_type = "training"
    config.wandb.name = "pickup_mini-sequential-sep_policies-child_p0.2-curriculum-ROB"

    config.log_path = "./logs/" + f"{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}"

    config.exp_group = "merge-validation"
    config.info_keywords = ("is_success", "overall_task_success", "active_task_level")

    return config
