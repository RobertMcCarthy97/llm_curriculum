from datetime import datetime
import numpy as np
import argparse

from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CallbackList, EveryNTimesteps
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback

import wandb
from wandb.integration.sb3 import WandbCallback

from env_wrappers import make_env, make_env_baseline
from stable_baselines3.common.buffers_custom import LLMBasicReplayBuffer
from sb3_callbacks import (
    SuccessCallback,
    VideoRecorderCallback,
    EvalCallbackMultiTask,
    GradientCallback,
)
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.custom_encoders import (
    CustomSimpleCombinedExtractor,
    FiLM,
    MixtureofExpertsEncoder,
    CAREDummy,
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
        )

    # Vec Env
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, info_keywords=hparams["info_keywords"])
    if hparams["norm_obs"]:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    else:
        assert False

    return env


def get_hparams():
    hparams = {
        "seed": 0,
        # env
        "manual_decompose_p": 1,
        "dense_rew_lowest": False,
        "dense_rew_tasks": [
            "move_gripper_to_cube",
            "move_cube_towards_target_grasp",
        ],  # ['move_gripper_to_cube', "move_cube_towards_target_grasp"]
        "use_language_goals": False,
        "render_mode": "rgb_array",
        "use_oracle_at_warmup": False,
        "max_ep_len": 50,
        "use_baseline_env": False,
        # task
        "single_task_names": [
            "move_gripper_to_cube",
            "cube_between_grippers",
            "close_gripper_cube",
            "lift_cube",
            "move_cube_towards_target_grasp",
        ],
        "high_level_task_names": ["move_cube_to_target"],
        "contained_sequence": False,
        # algo
        "algo": TD3,  # DDPG/TD3/SAC
        "policy_type": "MultiInputPolicy",
        "learning_starts": 1e3,
        "batch_size": 128,
        "action_noise": NormalActionNoise,  # NormalActionNoise, None
        "replay_buffer_class": None,  # LLMBasicReplayBuffer, None
        "replay_buffer_kwargs": None,  # None, {'keep_goals_same': True, 'do_parent_relabel': True, 'parent_relabel_p': 0.2}
        "total_timesteps": 1e5,
        "device": "cpu",
        "policy_kwargs": {
            "goal_based_custom_args": {"use_siren": False, "use_sigmoid": True}
        },  # {}, {'goal_based_custom_args': {'use_siren': True, 'use_sigmoid': True}}
        "gradient_steps": -1,
        "do_mtrl_hparam_boost": True,
        "norm_obs": True,
        # extractor
        "features_extractor": "care",  # None, 'film', 'custom', 'care', 'care_dummy
        "share_features_extractor": False,
        "CARE_n_experts": 2,
        ####
        # logging
        "do_track": True,
        "log_path": "./logs/" + f"{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}",
        "exp_name": "gripper2cube-betw-close-lift-2target-care2-sigmoid",
        "exp_group": "film+care-debug",
        "info_keywords": ("is_success", "overall_task_success", "active_task_level"),
    }
    # TODO: learning rates to 3e-4? (following MTRL)

    # Check for overrides
    hparams = override_hparams(hparams)

    ## Checks
    if hparams["use_language_goals"]:
        assert hparams["features_extractor"] is not None
    # features extractor
    hparams["features_extractor"] = get_features_extractor_hparams(hparams)
    if hparams["features_extractor"] is not None:
        assert (
            hparams["policy_type"] == "MultiInputPolicy"
        ), "not setup for other policy types yet"
        assert hparams["algo"] == TD3, "not setup for other algos yet"
        hparams["policy_kwargs"].update(hparams["features_extractor"])
    hparams["policy_kwargs"]["share_features_extractor"] = hparams[
        "share_features_extractor"
    ]

    return hparams


def get_features_extractor_hparams(hparams):
    extractor_name = hparams["features_extractor"]
    if extractor_name == None:
        return {}
    elif extractor_name == "film":
        return {"features_extractor_class": FiLM}
    elif extractor_name == "custom":
        return {
            "features_extractor_class": CustomSimpleCombinedExtractor,
            "features_extractor_kwargs": {
                "encoders": {"observation": "mlp", "desired_goal": "mlp"},
                "fuse_method": "concat",
            },
        }
    elif extractor_name == "care":
        return {
            "features_extractor_class": MixtureofExpertsEncoder,
            "features_extractor_kwargs": {
                "num_experts": hparams["CARE_n_experts"],
                "detach_emb_for_selection": True,
            },
        }
    elif extractor_name == "care_dummy":
        return {
            "features_extractor_class": CAREDummy,
            "features_extractor_kwargs": {"num_experts": hparams["CARE_n_experts"]},
        }
    else:
        assert False


def override_hparams(hparams):
    # Instantiate the parser
    parser = argparse.ArgumentParser(description="Hyperparameters for the experiment")

    # Check whether to override
    parser.add_argument("--override_hparams", action="store_true")

    # Add arguments to the parser for each hyperparameter
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--use_language_goals", action="store_true")
    parser.add_argument("--features_extractor", type=str)
    parser.add_argument("--share_features_extractor", action="store_true")
    parser.add_argument("--care_n_experts", type=int)

    # Parse the command line argumentsTrue
    args = parser.parse_args()

    if args.override_hparams:
        assert args.exp_name is not None
        assert "exp_name" in hparams
        hparams["exp_name"] = args.exp_name
        if args.use_language_goals is not None:
            assert "use_language_goals" in hparams
            hparams["use_language_goals"] = args.use_language_goals
        if args.features_extractor is not None:
            assert "features_extractor" in hparams
            hparams["features_extractor"] = args.features_extractor
        if args.share_features_extractor is not None:
            assert "share_features_extractor" in hparams
            hparams["share_features_extractor"] = args.share_features_extractor
        if args.care_n_experts is not None:
            assert "CARE_n_experts" in hparams
            hparams["CARE_n_experts"] = args.care_n_experts

    return hparams


"""
conda activate llm_curriculum
"""

if __name__ == "__main__":
    hparams = get_hparams()

    # W&B
    if hparams["do_track"]:
        run = wandb.init(
            entity="robertmccarthy11",
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
    eval_env = create_env(hparams)

    # multi-task hparams
    n_possible_tasks = len(env.envs[0].agent_conductor.get_possible_task_names())
    if n_possible_tasks > 1 and hparams["do_mtrl_hparam_boost"]:
        hparams["batch_size"] = hparams["batch_size"] * n_possible_tasks
        hparams["learning_starts"] = hparams["learning_starts"] * n_possible_tasks
        hparams["gradient_steps"] = 1 / n_possible_tasks
        hparams["policy_kwargs"]["net_arch"] = [400, 400, 400]

    # Logger and callbacks
    single_task_names = env.envs[0].agent_conductor.get_possible_task_names()
    base_freq = 1000
    log_freq = base_freq * max(1, len(single_task_names))  # TODO: make hparam
    vid_freq = base_freq * 10

    logger = configure(hparams["log_path"], ["stdout", "csv", "tensorboard"])

    callback_list = []
    callback_list += [
        EvalCallbackMultiTask(
            eval_env,
            eval_freq=log_freq,
            best_model_save_path=None,
            single_task_names=single_task_names,
        )
    ]  # TODO: use different eval env
    callback_list += [
        VideoRecorderCallback(
            eval_env, render_freq=vid_freq, n_eval_episodes=1, add_text=True
        )
    ]
    callback_list += [GradientCallback(log_freq=log_freq)]
    if not hparams["use_baseline_env"]:
        callback_list += [SuccessCallback(log_freq=log_freq)]
    # wandb
    if hparams["do_track"]:
        # log hyperparameters
        wandb.log({"hyperparameters": hparams})
        # wandb callback
        callback_list += [
            WandbCallback(
                gradient_save_freq=1000,
                model_save_path=None,
                verbose=1,
            )
        ]
    callback = CallbackList(callback_list)

    ### Algo
    # init action noise
    if hparams["action_noise"] is not None:
        assert hparams["action_noise"] is NormalActionNoise
        n_actions = env.action_space.shape[-1]
        hparams["action_noise"] = hparams["action_noise"](
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )  # TODO: make magnitude a hparam?
    # init model
    model = hparams["algo"](
        hparams["policy_type"],
        env,
        verbose=1,
        learning_starts=hparams["learning_starts"],
        batch_size=hparams["batch_size"],
        replay_buffer_class=hparams["replay_buffer_class"],
        replay_buffer_kwargs=hparams["replay_buffer_kwargs"],
        device=hparams["device"],
        use_oracle_at_warmup=hparams["use_oracle_at_warmup"],
        policy_kwargs=hparams["policy_kwargs"],
        action_noise=hparams["action_noise"],
        gradient_steps=hparams["gradient_steps"],
    )
    model.set_logger(logger)
    # import pdb; pdb.set_trace()

    # Train the model
    model.learn(
        total_timesteps=hparams["total_timesteps"],
        callback=callback,
        progress_bar=False,
    )

    if hparams["do_track"]:
        run.finish()
