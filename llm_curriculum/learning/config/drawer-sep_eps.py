from ml_collections import config_dict
from datetime import datetime
from llm_curriculum.envs.curriculum_manager import (
    SeperateEpisodesCM,
    DummySeperateEpisodesCM,
)
from stable_baselines3 import TD3
from stable_baselines3.common.buffers_custom import SeparatePoliciesReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise


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
    config.manual_decompose_p = 1
    config.dense_rew_lowest = False
    config.dense_rew_tasks = []
    config.use_language_goals = False
    config.render_mode = "rgb_array"
    config.oracle_at_warmup = {"use_oracle": True, "oracle_steps": 10e3}
    config.max_ep_len = 50
    config.use_baseline_env = False
    config.is_closed_on_reset = True
    config.cube_pos_on_reset = "table"
    # task
    config.single_task_names = ["move_cube_to_target"]
    config.high_level_task_names = ["move_cube_to_target"]
    config.curriculum_manager_cls = DummySeperateEpisodesCM  # DummySeperateEpisodesCM, SeperateEpisodesCM (CM decides 'decompose_p' based on success rates)
    config.sequenced_episodes = False
    config.contained_sequence = False
    config.initial_state_curriculum_p = 0.0
    config.child_p_strat = "mean"
    config.decompose_p_clip = {"low": 0.1, "high": 0.9}
    # algo
    config.algo = TD3
    config.policy_type = "MlpPolicy"
    config.learning_starts = 10e3
    config.replay_buffer_class = SeparatePoliciesReplayBuffer
    config.replay_buffer_kwargs = {"child_p": 0.2}
    config.only_use_nearest_children_data = False
    config.total_timesteps = 1e6
    config.device = "cpu"
    config.policy_kwargs = None  # None, {'goal_based_custom_args': {'use_siren': True, 'use_sigmoid': True}}
    config.action_noise = NormalActionNoise
    config.batch_size = 100
    # TODO: increase batch size??
    # logging
    config.wandb = config_dict.ConfigDict()
    config.wandb.track = True
    config.wandb.project = "llm-curriculum"
    config.wandb.entity = "robertmccarthy11"
    config.wandb.group = "drawer-env-testing"
    config.wandb.job_type = "training"
    config.wandb.name = "move_cube_to_target-single-oracle_10k"

    config.log_path = "./logs/" + f"{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}"
    config.save_models = False
    config.eval_policy = True
    config.eval_traversal_modes = ["train"]

    config.exp_group = "merge-validation"
    config.info_keywords = ("is_success", "overall_task_success", "active_task_level")

    # assert False, "increase episode length!"

    return config
