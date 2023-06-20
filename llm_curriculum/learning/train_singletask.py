import argparse
import numpy as np
import pathlib

from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from llm_curriculum.envs import make_single_task_env
from llm_curriculum.envs.wrappers import NonGoalNonDictObsWrapper
from llm_curriculum.learning import utils


def add_algo_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument(
        "--eval-frequency", type=int, default=10000, help="Evaluate agent every N steps"
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=10000,
        help="Save agent every N steps",
    )
    parser.add_argument(
        "--video-frequency",
        type=int,
        default=100000,
    )
    parser.add_argument("--eval-episodes", type=int, default=5)

    return parser


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser = utils.add_basic_args(parser)
    parser = add_algo_args(parser)
    args = parser.parse_args()
    run_name = utils.make_run_name(args)
    log_path = f"./logs/{run_name}"
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    # Set up environment
    env = make_single_task_env(
        args.env_id, render_mode="rgb_array" if args.capture_video else None
    )
    env = NonGoalNonDictObsWrapper(env)
    env = Monitor(env, log_path + "/monitor.csv")
    env = DummyVecEnv([lambda: env])

    if args.capture_video:
        env = VecVideoRecorder(
            env,
            log_path + "/videos",
            record_video_trigger=lambda x: x % args.video_frequency == 0,
        )
    eval_env = env

    # Set up logging
    callback_list = []
    callback_list.append(
        EvalCallback(
            eval_env,
            best_model_save_path=log_path,  # folder to save best_model.zip
            log_path=log_path,  # folder to save evaluations.npz
            eval_freq=args.eval_frequency,  # evaluate the agent every eval_freq timesteps
            deterministic=True,
            render=args.capture_video,
        )
    )

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group_name,
            name=run_name,
            job_type="training",
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

        from wandb.integration.sb3 import WandbCallback

        callback_list.append(
            # Save current model to WandB
            WandbCallback(
                model_save_freq=args.checkpoint_frequency,
                model_save_path=log_path,
                verbose=1,
            )
        )

    callback = CallbackList(callback_list)

    # Set up model
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )
    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log=log_path,
        device="cuda" if args.cuda else "cpu",
    )

    # Start training
    model.learn(
        total_timesteps=args.total_timesteps, log_interval=10, callback=callback
    )

    # Save best model to WandB
    if args.track:
        wandb.save(log_path + "/best_model.zip")
