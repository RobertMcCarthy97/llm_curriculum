""" Pre-train a TD3 agent by bootstrapping from other TD3 agents' critics."""
import argparse
import numpy as np
import pathlib
import gymnasium as gym
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.policies import ContinuousCritic

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
    parser.add_argument("--pretrain-checkpoint-A", type=str, required=True)
    parser.add_argument("--pretrain-checkpoint-B", type=str, required=True)
    return parser


def get_q_value(
    critic: ContinuousCritic, obs: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    """Get the q value of a critic for a given observation and action."""
    ensemble_q_value = critic(obs, action)
    ensemble_q_value = torch.hstack(ensemble_q_value)
    q_value = torch.min(ensemble_q_value, dim=-1, keepdim=True)[0]
    return q_value


def pretrain_critic(
    pretrain_A: ContinuousCritic,
    pretrain_B: ContinuousCritic,
    model_critic: ContinuousCritic,
    observation_space: gym.spaces.Box,
    action_space: gym.spaces.Box,
    # Training params
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    n_iters: int = 100,
):
    """Pretrain the critic of a TD3 agent by bootstrapping from other TD3 agents' critics."""
    model_critic.train()
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model_critic.parameters(), lr=learning_rate)

    for train_iter in range(n_iters):
        # Sample batch of observations and actions
        # TODO: vectorize sampling code
        observations = torch.zeros(
            (batch_size, observation_space.shape[0]), device=model_critic.device
        )
        actions = torch.zeros(
            (batch_size, action_space.shape[0]), device=model_critic.device
        )
        for i in range(batch_size):
            obs = observation_space.sample()
            action = action_space.sample()
            observations[i] = torch.tensor(obs)
            actions[i] = torch.tensor(action)

        with torch.no_grad():
            Q_A = get_q_value(pretrain_A, observations, actions)
            Q_B = get_q_value(pretrain_B, observations, actions)
            # TODO: Is this correct?
            # TODO: Constrain Q_A, Q_B to be non-positive?
            target_Q = torch.logsumexp(
                torch.cat([Q_A, Q_A + Q_B], dim=-1), dim=-1, keepdim=True
            ) + torch.log(torch.tensor(0.5))

        print(target_Q.mean())
        pred_Qs = model_critic(observations, actions)
        optim.zero_grad()
        loss = 0
        for pred_Q in pred_Qs:
            assert pred_Q.shape == target_Q.shape
            loss += loss_fn(pred_Q, target_Q)
        loss.backward()
        optim.step()
        print("Iter: {}, Loss: {}".format(train_iter, loss.item()))


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

    pretrain_A = TD3.load(args.pretrain_checkpoint_A)
    pretrain_B = TD3.load(args.pretrain_checkpoint_B)

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

    critic_A = pretrain_A.critic
    critic_B = pretrain_B.critic
    model_critic = model.critic

    # TODO: Implement training loop for model critic
    pretrain_critic(
        critic_A, critic_B, model_critic, env.observation_space, env.action_space
    )
