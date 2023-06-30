from ml_collections import config_dict


def get_env_config():
    """Returns the default config"""
    # TODO: Refactor get_config to use this function
    config = config_dict.ConfigDict()

    config.help = False
    config.exp_name = "td3_unlockpickup"
    config.device = "cpu"
    config.total_timesteps = 1e6

    config.env = config_dict.ConfigDict()
    config.env.id = "MiniGrid-UnlockPickup-v0"

    # Logging
    config.wandb = config_dict.ConfigDict()
    config.wandb.track = False
    config.wandb.entity = "ucl-air-lab"
    config.wandb.project = "llm-curriculum"
    config.wandb.group = "minigrid"
    config.wandb.job_type = "training"

    return config
