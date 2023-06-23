""" Run vanilla RL on the FetchPickAndPlace task. """

import wandb

from typing import Any, Dict
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


def training_loop(models_dict, env, hparams, log_interval=4, callback=None):

    total_timesteps = hparams.total_timesteps
    for task_name in hparams.high_level_task_names:
        assert len(env.envs) == 1

        # Get model
        model = models_dict[task_name]

        # Set task
        for vec_env in env.envs:
            vec_env.agent_conductor.set_single_task_names([task_name])

        # Collect data
        rollout = model.collect_rollouts(
            env,
            train_freq=model.train_freq,
            action_noise=model.action_noise,
            callback=callback,
            learning_starts=model.learning_starts,
            replay_buffer=model.replay_buffer,
            log_interval=log_interval,
            reset_b4_collect=True,
        )
        # TODO: collect 2 rollouts each to reduce cost of the extra rollout reset?

        if model.num_timesteps > 0 and model.num_timesteps > model.learning_starts:
            # If no `gradient_steps` is specified,
            # do as many gradients steps as steps performed during the rollout
            gradient_steps = (
                model.gradient_steps
                if model.gradient_steps >= 0
                else rollout.episode_timesteps
            )
            # Special case when the user passes `gradient_steps=0`
            if gradient_steps > 0:
                model.train(batch_size=model.batch_size, gradient_steps=gradient_steps)

        # end condition
        if model.num_timesteps > total_timesteps:  # TODO: use global env count instead
            break


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

    # Seed
    set_random_seed(hparams["seed"])  # type:ignore

    # create envs
    env: "VecNormalize" = create_env(hparams)

    # setup logging
    logger, callback = setup_logging(hparams, env)
    assert len(env.envs) == 1
    env.envs[0].agent_conductor.set_logger(logger)

    # create models
    models_dict: Dict[str, Any] = create_models(env, logger, hparams)

    # Only keep the model for the high-level task
    assert hparams.high_level_task_names is not None
    assert len(hparams.high_level_task_names) == 1
    task_name = hparams.high_level_task_names[0]
    models_dict = {task_name: models_dict[task_name]}

    # Train
    init_training(models_dict, hparams["total_timesteps"], callback=callback)
    training_loop(models_dict, env, hparams, callback=callback)

    # Close
    if hparams.wandb.track:
        run.finish()


if __name__ == "__main__":
    app.run(main)
