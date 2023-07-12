from datetime import datetime
import numpy as np
import copy
import os

# from stable_baselines3 import TD3
from llm_curriculum.learning.sb3.td3_custom import TD3
from stable_baselines3.common.logger import configure, HumanOutputFormat
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.type_aliases import (
    MaybeCallback,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.noise import NormalActionNoise


import wandb
from wandb.integration.sb3 import WandbCallback

from llm_curriculum.envs.make_env import make_env, make_env_baseline
from llm_curriculum.envs.curriculum_manager import (
    SeperateEpisodesCM,
    DummySeperateEpisodesCM,
)
from llm_curriculum.learning.sb3.sequenced_rollouts import SequencedRolloutCollector

from llm_curriculum.learning.sb3.callback import (
    VideoRecorderCallback,
    EvalCallbackMultiTask,
    SuccessCallbackSeperatePolicies,
)

from absl import app
from absl import flags
from ml_collections import config_flags

_CONFIG = config_flags.DEFINE_config_file("config", None)
flags.mark_flag_as_required("config")


def create_env(hparams, eval=False, vec_norm_path=None):
    hparams = copy.deepcopy(hparams)
    if eval:
        hparams["initial_state_curriculum_p"] = 0.0
    # Create env
    if hparams["use_baseline_env"]:
        env = make_env_baseline(
            "FetchReach-v2",
            render_mode=hparams["render_mode"],
            max_ep_len=hparams["max_ep_len"],
        )
        hparams["replay_buffer_class"] = None
        hparams["info_keywords"] = ("is_success",)
    else:
        env = make_env(
            drawer_env=hparams["drawer_env"],
            manual_decompose_p=hparams["manual_decompose_p"],
            dense_rew_lowest=hparams["dense_rew_lowest"],
            dense_rew_tasks=hparams["dense_rew_tasks"],
            use_language_goals=hparams["use_language_goals"],
            render_mode=hparams["render_mode"],
            max_ep_len=hparams["max_ep_len"],
            single_task_names=hparams["single_task_names"],
            high_level_task_names=hparams["high_level_task_names"],
            contained_sequence=hparams["contained_sequence"],
            state_obs_only=True,
            curriculum_manager_cls=hparams["curriculum_manager_cls"],
            use_incremental_reward=hparams["incremental_reward"],
            initial_state_curriculum_p=hparams["initial_state_curriculum_p"],
            is_closed_on_reset=hparams["is_closed_on_reset"],
            cube_pos_on_reset=hparams["cube_pos_on_reset"],
            child_p_strat=hparams["child_p_strat"],
            decompose_p_clip=hparams["decompose_p_clip"],
            task_complete_thresh=hparams["task_complete_thresh"],
        )

    # Vec Env
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, info_keywords=hparams["info_keywords"])
    if vec_norm_path is None:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    else:
        env = VecNormalize.load(vec_norm_path, env)

    return env


def setup_logging(hparams, train_env, base_freq=1000):

    single_task_names = train_env.envs[0].agent_conductor.get_possible_task_names()
    log_freq = base_freq * max(1, len(single_task_names))  # TODO: make hparam
    vid_freq = base_freq * 10

    # Logger and callbacks
    logger = configure(hparams["log_path"], ["stdout", "csv", "tensorboard"])
    for output_format in logger.output_formats:
        if isinstance(output_format, HumanOutputFormat):
            output_format.max_length = 50

    # create eval envs
    if hparams["sequenced_episodes"]:
        eval_env_sequenced = create_env(hparams)
        non_seq_params = copy.deepcopy(hparams)
        non_seq_params.update(
            {
                "sequenced_episodes": False,
                "contained_sequence": False,
                "single_task_names": single_task_names,
                "manual_decompose_p": 1,
            }
        )
        eval_env_non_seq = create_env(non_seq_params)
        video_env = eval_env_sequenced
    else:
        eval_env_non_seq = create_env(hparams, eval=True)
        eval_env_sequenced = None
        video_env = eval_env_non_seq
    # TODO: link train curriculum manager + agent_conductor to eval envs?? (so can get same decompositions in eval...)

    callback_list = []
    if hparams["do_seperate_policy_eval"] or len(hparams["eval_traversal_modes"]) > 0:
        callback_list += [
            EvalCallbackMultiTask(
                eval_env_non_seq,
                eval_env_sequenced=eval_env_sequenced,
                eval_freq=log_freq,
                best_model_save_path=None,
                seperate_policies=True,
                single_task_names=single_task_names,
                tree_traversal_modes=hparams["eval_traversal_modes"],
                do_seperate_policy_eval=hparams["do_seperate_policy_eval"],
                n_eval_episodes=10,
            )
        ]
    if not hparams["use_baseline_env"]:
        callback_list += [
            SuccessCallbackSeperatePolicies(
                log_freq=log_freq,
                do_save_models=hparams["save_models"],
                save_dir=os.path.join("./models", hparams.wandb.name),
                single_task_names=single_task_names,
            )
        ]
    callback_list += [
        VideoRecorderCallback(
            video_env,
            render_freq=vid_freq,
            n_eval_episodes=1,
            add_text=True,
            sequenced_rollouts=hparams["sequenced_episodes"],
        )
    ]
    # wandb
    if hparams.wandb.track:
        # log hyperparameters
        wandb.log({"hyperparameters": hparams.to_dict()})
        # wandb callback
        callback_list += [
            WandbCallback(
                gradient_save_freq=10,
                model_save_path=None,
                verbose=1,
            )
        ]
    callback = CallbackList(callback_list)

    return logger, callback


def create_models(env, logger, hparams):
    task_list = env.envs[0].agent_conductor.get_possible_task_names()
    assert len(task_list) > 0

    # init action noise
    if hparams["action_noise"] is not None:
        assert hparams["action_noise"] is NormalActionNoise
        n_actions = env.action_space.shape[-1]
        hparams["action_noise"] = hparams["action_noise"](
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )  # TODO: make magnitude a hparam?

    models_dict = {}
    # create models
    for task_name in task_list:
        assert hparams["algo"] is TD3
        model = hparams["algo"](
            hparams["policy_type"],
            env,
            verbose=1,
            learning_starts=hparams["learning_starts"],
            replay_buffer_class=hparams["replay_buffer_class"],
            replay_buffer_kwargs=hparams["replay_buffer_kwargs"].to_dict(),
            device=hparams["device"],
            oracle_at_warmup=hparams["oracle_at_warmup"],
            policy_kwargs=hparams["policy_kwargs"],
            action_noise=hparams["action_noise"],
            batch_size=hparams["batch_size"],
            task_name=task_name,
        )
        model.set_logger(logger)
        if hparams["replay_buffer_class"] is not None:
            model.replay_buffer.set_task_name(task_name)
        models_dict[task_name] = model

    if hparams["replay_buffer_class"] is not None:
        # link buffers relations
        for task_name in models_dict.keys():
            assert len(env.envs) == 1
            relations = (
                env.envs[0]
                .agent_conductor.get_task_from_name(task_name)
                .get_relations(nearest=hparams["only_use_nearest_children_data"])
            )
            models_dict[task_name].replay_buffer.init_datasharing(
                relations, models_dict, agent_conductor=env.envs[0].agent_conductor
            )
        # TODO: print / save these relations to ensure all correct...

    return models_dict


def init_training(
    models_dict,
    total_timesteps: int,
    callback: MaybeCallback = None,
    tb_log_name: str = "run",
    reset_num_timesteps: bool = True,
    progress_bar: bool = False,
):

    for _, model in models_dict.items():
        total_timesteps, callback = model._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        callback.on_training_start(locals(), globals())


def training_loop(models_dict, env, total_timesteps, log_interval=4, callback=None):

    for task_name in env.envs[0].agent_conductor.curriculum_manager:
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

        # Train
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


def training_loop_sequential(
    models_dict,
    env,
    total_timesteps,
    logger,
    log_interval=4,
    callback=None,
    save_freq=20000,
):  # TODO: log interval

    rollout_collector = SequencedRolloutCollector(env, models_dict)
    timesteps_count = 0
    save_after = save_freq

    while timesteps_count < total_timesteps:
        # Collect data
        rollout, models_steps_taken = rollout_collector.collect_rollouts(
            # env,
            train_freq=TrainFreq(
                1, TrainFrequencyUnit.EPISODE
            ),  # TODO: make hyperparam
            # action_noise=model.action_noise, # TODO: wopuld action noise be beneficial??
            callback=callback,
            # learning_starts=model.learning_starts,
            # replay_buffer=model.replay_buffer,
            log_interval=log_interval,
        )
        # TODO: collect 2 rollouts each to reduce cost of the extra rollout reset?
        timesteps_count += rollout.episode_timesteps
        logger.record("time/train_loop_timesteps", timesteps_count)

        # Train
        for model_name, model in models_dict.items():
            if model.num_timesteps > 0 and model.num_timesteps > model.learning_starts:
                ###### Custom: we just do 1 grad step per env step
                assert model.gradient_steps < 0
                gradient_steps = models_steps_taken[model_name]
                if gradient_steps > 0:
                    model.train(
                        batch_size=model.batch_size, gradient_steps=gradient_steps
                    )


def get_hparams():

    hparams = _CONFIG.value

    ##### Checks
    if hparams["contained_sequence"]:
        assert hparams["sequenced_episodes"]
    # if hparams['sequenced_episodes']:
    #     assert len(hparams['single_task_names']) == 0 or hparams['contained_sequence']
    if hparams["policy_kwargs"] is not None:
        assert hparams["policy_type"] == "MlpPolicy"
        assert hparams["algo"] == TD3
    if hparams["action_noise"] is not None:
        assert hparams["algo"] == TD3

    if hparams["curriculum_manager_cls"] is not None:
        if hparams["manual_decompose_p"] is not None:
            assert not isinstance(
                hparams["curriculum_manager_cls"], SeperateEpisodesCM
            ), "manual decompose_p overrides CM"

    if hparams["save_models"]:
        assert hparams["sequenced_episodes"], "not setup for non-sequenced episodes"

    if hparams["only_use_nearest_children_data"]:
        assert hparams["child_p_strat"] == "sequenced_direct_children"

    if hparams["replay_buffer_kwargs"]["parent_child_split"]["strat"] == "all_success":
        assert hparams[
            "sequenced_episodes"
        ]  # exploit_sequenced_success rates only work if doing full tree learning...

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
    else:
        run = None

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

    # Train
    init_training(models_dict, hparams["total_timesteps"], callback=callback)
    if hparams["sequenced_episodes"]:
        training_loop_sequential(
            models_dict,
            env,
            hparams["total_timesteps"],
            logger,
            callback=callback,
        )
    else:
        training_loop(models_dict, env, hparams["total_timesteps"], callback=callback)

    if hparams["do_track"]:
        run.finish()


if __name__ == "__main__":
    app.run(main)
