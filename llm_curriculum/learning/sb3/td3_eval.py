import argparse
import pathlib

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy

from llm_curriculum.envs import make_single_task_env
from llm_curriculum.envs.wrappers import NonGoalNonDictObsWrapper
from llm_curriculum.learning import utils

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser = utils.add_basic_args(parser)
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
            env, log_path + "/videos", record_video_trigger=lambda x: True
        )
    eval_env = env

    model = TD3.load(f"downloads/{args.env_id}/model.zip")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
