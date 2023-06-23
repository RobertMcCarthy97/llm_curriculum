from ml_collections import config_dict
from datetime import datetime

from stable_baselines3 import TD3
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
    config.use_baseline_env = True
    config.render_mode = "rgb_array"
    config.max_ep_len = 50
    config.info_keywords = ("is_success",)

    # algo
    config.algo = TD3
    config.policy_type = "MultiInputPolicy"
    config.learning_starts = 1e3
    config.use_her = False
    config.total_timesteps = 1e6
    config.device = "cpu"

    # logging
    config.wandb = config_dict.ConfigDict()
    config.wandb.track = False
    config.wandb.project = "llm-curriculum"
    config.wandb.entity = "ucl-air-lab"
    config.wandb.group = "baselines"
    config.wandb.job_type = "training"
    config.wandb.name = "baseline"

    config.log_path = "./logs/" + f"{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}"

    return config
