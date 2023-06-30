import numpy as np
import pathlib
import minigrid  # noqa
import gymnasium as gym

from absl import app
from absl import flags
from ml_collections import config_flags

from stable_baselines3 import TD3
from llm_curriculum.envs.minigrid.env_wrapper import MinigridTaskEnvWrapper

_CONFIG = config_flags.DEFINE_config_file("config", None)
flags.mark_flag_as_required("config")


def main(argv):
    # Parse arguments

    config = _CONFIG.value
    log_path = f"./logs/{config.exp_name}"

    print(config)
    if config.help:
        # Exit after printing config
        return

    env = gym.make(config.env.id)
    if config.env.subtasks:
        env = MinigridTaskEnvWrapper(env, config.env.subtasks)

    if config.wandb.track:
        import wandb

        run = wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            group=config.wandb.group,
            name=config.exp_name,
            job_type=config.wandb.job_type,
            # config=vargs,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=False,  # auto-upload the videos of agents playing the game
            save_code=False,  # optional
        )

    # Set up model
    model = TD3(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_path,
        device=config.device,
    )

    # Start training
    model.learn(total_timesteps=config.total_timesteps, log_interval=10)


if __name__ == "__main__":
    app.run(main)
