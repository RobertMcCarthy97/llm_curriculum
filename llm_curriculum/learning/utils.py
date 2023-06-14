import gym
import time
import argparse

from distutils.util import strtobool

def make_run_name(args):
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    return run_name

def add_basic_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """ Universal arguments """
    # fmt: off
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-group-name", type=str, default=None,
        help="the wandb's group name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    return parser

def maybe_wrap_env_video(env: gym.Env, capture_video: bool, run_name: str):
    """ Wraps gym environment with a video recorder if necessary """
    if capture_video:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    return env

def wrap_env(env: gym.Env, idx: int, capture_video: bool, run_name: str):
    """ Wraps gym environment with some basic wrappers """
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video and idx == 0:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    return env