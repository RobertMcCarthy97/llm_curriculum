""" Run vanilla RL on the FetchPickAndPlace task. """

import wandb
import gymnasium as gym

from typing import Any, Dict
from absl import app
from absl import flags
from ml_collections import config_flags
from llm_curriculum.learning.utils import (
    make_env_baseline,
    set_random_seed,
    maybe_create_wandb_callback,
)

from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from llm_curriculum.envs.wrappers import NonGoalNonDictObsWrapper

_CONFIG = config_flags.DEFINE_config_file("config", None)
flags.mark_flag_as_required("config")


def get_hparams():
    hparams = _CONFIG.value
    return hparams


def setup_logging(hparams, env):
    # Logger and callbacks
    logger = configure(hparams.log_path, ["stdout", "csv", "tensorboard"])
    return logger


def create_env(hparams):
    env = make_env_baseline(
        "FetchPickAndPlace-v2",
        render_mode=hparams.render_mode,
        max_ep_len=hparams.max_ep_len,
    )
    env = NonGoalNonDictObsWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, info_keywords=hparams.info_keywords)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return env


def create_model(hparams, env):
    model = hparams.algo(
        hparams.policy_type,
        env,
        verbose=1,
        learning_starts=hparams.learning_starts,
        device=hparams.device,
    )
    return model


def main(argv):
    """
    TODO: implement W&B sweep?
    """
    hparams: "ml_collections.config_dict.ConfigDict" = get_hparams()
    print(hparams)
    if hparams["help"]:
        # Exit after printing hparams
        return

    # W&B
    if hparams.wandb.track:
        run = wandb.init(
            entity=hparams.wandb.entity,
            project=hparams.wandb.project,
            group=hparams.wandb.group,
            name=hparams.wandb.name,
            job_type=hparams.wandb.job_type,
            # config=vargs,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=False,  # auto-upload the videos of agents playing the game
            save_code=False,  # optional
        )
        # log hyperparameters
        wandb.log({"hyperparameters": hparams})

    # Seed
    set_random_seed(hparams["seed"])  # type:ignore

    # create envs
    env = create_env(hparams)
    model = create_model(hparams, env)

    # logging
    logger = setup_logging(hparams, env)
    model.set_logger(logger)

    # Callbacks
    callback_list = []
    callback_list += maybe_create_wandb_callback(hparams, env)
    callback = CallbackList(callback_list)

    # Train
    model.learn(
        total_timesteps=hparams.total_timesteps,
        callback=callback,
        log_interval=4,
    )

    # Close
    if hparams.wandb.track:
        run.finish()


if __name__ == "__main__":
    app.run(main)
