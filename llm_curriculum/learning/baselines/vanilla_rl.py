""" Run vanilla RL on the FetchPickAndPlace task. """

import wandb

from absl import app
from absl import flags
from ml_collections import config_flags
from llm_curriculum.learning.utils import (
    create_env,
    create_models,
    init_training,
    setup_logging,
    training_loop,
    check_hparams,
    set_random_seed,
)

_CONFIG = config_flags.DEFINE_config_file("config", None)
flags.mark_flag_as_required("config")


def get_hparams():
    hparams = _CONFIG.value
    check_hparams(hparams)
    return hparams


def main(argv):
    """
    TODO: implement W&B sweep?
    """
    hparams = get_hparams()
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

    # Seed
    set_random_seed(hparams["seed"])  # type:ignore

    # create envs
    env = create_env(hparams)

    # setup logging
    logger, callback = setup_logging(hparams, env)
    assert len(env.envs) == 1
    env.envs[0].agent_conductor.set_logger(logger)

    # create models
    models_dict = create_models(env, logger, hparams)

    # Only keep the model for the high-level task
    assert hparams.high_level_task_names is not None
    assert len(hparams.high_level_task_names) == 1
    task_name = hparams.high_level_task_names[0]
    models_dict = {task_name: models_dict[task_name]}

    # Train
    init_training(models_dict, hparams["total_timesteps"], callback=callback)
    training_loop(models_dict, env, hparams["total_timesteps"], callback=callback)

    # Close
    if hparams.wandb.track:
        run.finish()


if __name__ == "__main__":
    app.run(main)
