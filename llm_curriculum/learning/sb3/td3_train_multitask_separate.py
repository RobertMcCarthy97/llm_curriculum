from datetime import datetime
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure
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
from llm_curriculum.envs.curriculum_manager import SeperateEpisodesCM
from llm_curriculum.envs.sequenced_rollouts import SequencedRolloutCollector

from stable_baselines3.common.buffers_custom import (
    SeparatePoliciesReplayBuffer,
)  # TODO: should move these into this repo!!??
from llm_curriculum.learning.sb3.callback import (
    VideoRecorderCallback,
    EvalCallbackMultiTask,
    SuccessCallbackSeperatePolicies,
)


def create_env(hparams):
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
        )

    # Vec Env
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, info_keywords=hparams["info_keywords"])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    return env


def setup_logging(hparams, train_env, base_freq=1000):

    single_task_names = train_env.envs[0].agent_conductor.get_possible_task_names()
    log_freq = base_freq * max(1, len(single_task_names))  # TODO: make hparam
    vid_freq = base_freq * 10

    # Logger and callbacks
    logger = configure(hparams["log_path"], ["stdout", "csv", "tensorboard"])

    # create eval envs
    if hparams["sequenced_episodes"]:
        eval_env_sequenced = create_env(hparams)
        non_seq_params = hparams.copy()
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
        eval_env_non_seq = create_env(hparams)
        eval_env_sequenced = None
        video_env = eval_env_non_seq
    # TODO: link train curriculum manager + agent_conductor to eval envs?? (so can get same decompositions in eval...)

    callback_list = []
    callback_list += [
        EvalCallbackMultiTask(
            eval_env_non_seq,
            eval_env_sequenced=eval_env_sequenced,
            eval_freq=log_freq,
            best_model_save_path=None,
            seperate_policies=True,
            single_task_names=single_task_names,
        )
    ]
    if not hparams["use_baseline_env"]:
        callback_list += [SuccessCallbackSeperatePolicies(log_freq=log_freq)]
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
    if hparams["do_track"]:
        # log hyperparameters
        wandb.log({"hyperparameters": hparams})
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
    for task in task_list:
        model = hparams["algo"](
            hparams["policy_type"],
            env,
            verbose=1,
            learning_starts=hparams["learning_starts"],
            replay_buffer_class=hparams["replay_buffer_class"],
            replay_buffer_kwargs=hparams["replay_buffer_kwargs"],
            device=hparams["device"],
            use_oracle_at_warmup=hparams["use_oracle_at_warmup"],
            policy_kwargs=hparams["policy_kwargs"],
            action_noise=hparams["action_noise"],
        )
        model.set_logger(logger)
        if hparams["replay_buffer_class"] is not None:
            model.replay_buffer.set_task_name(task)
        models_dict[task] = model

    if hparams["replay_buffer_class"] is not None:
        # link buffers relations
        for task_name in models_dict.keys():
            assert len(env.envs) == 1
            relations = (
                env.envs[0]
                .agent_conductor.get_task_from_name(task_name)
                .get_relations()
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


def training_loop(models_dict, env, total_timesteps, log_interval=4):

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
    models_dict, env, total_timesteps, logger, log_interval=4
):  # TODO: log interval

    rollout_collector = SequencedRolloutCollector(env, models_dict)
    timesteps_count = 0

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
    hparams = {
        "seed": 0,
        # env
        "manual_decompose_p": 1,
        "dense_rew_lowest": False,
        "dense_rew_tasks": [],  #
        "use_language_goals": False,
        "render_mode": "rgb_array",
        "use_oracle_at_warmup": False,  #
        "max_ep_len": 50,
        "use_baseline_env": False,
        # task
        "single_task_names": ["move_gripper_to_cube"],  #
        "high_level_task_names": ["move_cube_to_target"],
        "curriculum_manager_cls": SeperateEpisodesCM,  # DummySeperateEpisodesCM, SeperateEpisodesCM
        "sequenced_episodes": False,
        "contained_sequence": False,
        # algo
        "algo": TD3,  # DDPG/TD3/SAC
        "policy_type": "MlpPolicy",  # "MlpPolicy", "MultiInputPolicy"
        "learning_starts": 1e3,
        "replay_buffer_class": SeparatePoliciesReplayBuffer,  # LLMBasicReplayBuffer, None, SeparatePoliciesReplayBuffer
        "replay_buffer_kwargs": {
            "child_p": 0.2
        },  # None, {'keep_goals_same': True, 'do_parent_relabel': True, 'parent_relabel_p': 0.2}, {'child_p': 0.2}
        "total_timesteps": 1e6,
        "device": "cpu",
        "policy_kwargs": None,  # None, {'goal_based_custom_args': {'use_siren': True, 'use_sigmoid': True}}
        "action_noise": NormalActionNoise,  # NormalActionNoise, None
        # logging
        "do_track": True,
        "log_path": "./logs/" + f"{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}",
        "exp_name": "move_gripper_to_cube-only-seperate_code-NO_action_noise1",
        "exp_group": "sequential-iter_amp",
        "info_keywords": ("is_success", "overall_task_success", "active_task_level"),
    }

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

    return hparams


if __name__ == "__main__":
    """
    TODO: implement W&B sweep?
    """

    hparams = get_hparams()

    # W&B
    if hparams["do_track"]:
        run = wandb.init(
            entity="ucl-air-lab",
            project="llm-curriculum",
            group=hparams["exp_group"],
            name=hparams["exp_name"],
            job_type="training",
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

    # Train
    init_training(models_dict, hparams["total_timesteps"], callback=callback)
    if hparams["sequenced_episodes"]:
        training_loop_sequential(models_dict, env, hparams["total_timesteps"], logger)
    else:
        training_loop(models_dict, env, hparams["total_timesteps"])

    if hparams["do_track"]:
        run.finish()
