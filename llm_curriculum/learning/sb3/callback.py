from typing import Any, Dict, Optional, Union, List
import os
import warnings

import gym
import torch as th
import numpy as np
from PIL import Image, ImageDraw

from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    sync_envs_normalization,
)

from llm_curriculum.learning.sb3.sequenced_rollouts import evaluate_sequenced_policy


class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        render_freq: int,
        n_eval_episodes: int = 1,
        add_text: bool = False,
        sequenced_rollouts: bool = False,
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self.add_text = add_text
        self.sequenced_rollouts = sequenced_rollouts

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            self._record_video(deterministic=True)
            self._record_video(deterministic=False)
        return True

    def _record_video(self, deterministic=True):
        screens = []
        behaviour_str = "deterministic" if deterministic else "stochastic"

        def overlay_text(rgb_array: np.array, info: Dict[str, Any] = None):
            # import pdb; pdb.set_trace()
            # Convert the RGB array to a PIL image
            img = Image.fromarray(rgb_array.astype(np.uint8))
            # Call draw Method to add 2D graphics in an image
            I1 = ImageDraw.Draw(img)
            # Add Text to an image
            if "active_task_name" in info:
                I1.text(
                    (28, 36),
                    f"Active task: {info['active_task_name']}",
                    fill=(255, 0, 0),
                )
            I1.text((28, 50), f"Success: {info['is_success']}", fill=(255, 0, 0))
            # Convert the PIL image back to an RGB array
            rgb_array_with_text = np.array(img).astype(np.uint8)
            return rgb_array_with_text

        def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
            """
            Renders the environment in its current state, recording the screen in the captured `screens` list

            :param _locals: A dictionary containing all local variables of the callback's scope
            :param _globals: A dictionary containing all global variables of the callback's scope
            """
            screen = self._eval_env.render(mode="rgb_array")
            if self.add_text:
                assert len(self._eval_env.envs) == 1, "not setup for more than 1 env!"
                screen = overlay_text(screen, _locals["infos"][0])
            # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
            screens.append(screen.transpose(2, 0, 1))

        if self.sequenced_rollouts:
            raise NotImplementedError()
        else:
            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=deterministic,
            )
        self.logger.record(
            f"trajectory/video_{behaviour_str}",
            Video(th.ByteTensor(np.array([screens])), fps=25),
            exclude=("stdout", "log", "json", "csv"),
        )


class SuccessCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        # Dump
        if self.num_timesteps % self.log_freq == 0:
            assert len(self.locals["env"].envs) == 1, "not setup for more than 1 env!"
            env = self.locals["env"].envs[0]
            stats = env.agent_conductor.get_stats()
            # record averages
            for stat_key in stats.keys():
                for time_key in stats[stat_key].keys():
                    for task_key in stats[stat_key][time_key].keys():
                        self.logger.record(
                            f"custom_rollout_{stat_key}_{time_key}/{task_key}",
                            stats[stat_key][time_key][task_key],
                        )
            # record timestep
            self.logger.record("time/custom_timestep", self.num_timesteps)
            self.logger.record(
                "time/custom_timestep_multi_run",
                self.num_timesteps / len(env.agent_conductor.get_possible_task_names()),
            )  # just divide by n_tasks (assumes sampled equally)
            # dump and reset
            self.logger.dump(self.num_timesteps)
            env.agent_conductor.reset_epoch_stats()

        return True


class SuccessCallbackSeperatePolicies(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.num_timesteps_multi_run = 0

    def _on_step(self) -> bool:
        # Dump
        if self.num_timesteps_multi_run % self.log_freq == 0:
            assert len(self.locals["env"].envs) == 1, "not setup for more than 1 env!"
            env = self.locals["env"].envs[0]

            # record average stats from agent conductor
            stats = env.agent_conductor.get_stats()
            for stat_key in stats.keys():
                for time_key in stats[stat_key].keys():
                    for task_key in stats[stat_key][time_key].keys():
                        self.logger.record(
                            f"custom_rollout_{stat_key}_{time_key}/{task_key}",
                            stats[stat_key][time_key][task_key],
                        )

            # record timesteps
            self.logger.record(
                "time/custom_timestep",
                self.num_timesteps_multi_run / len(self.locals["models_dict"].keys()),
            )  # average per task
            self.logger.record(
                "time/custom_timestep_multi-run", self.num_timesteps_multi_run
            )  # overall
            for task_key, model in self.locals["models_dict"].items():
                self.logger.record(
                    f"time/{task_key}_steps", model.num_timesteps
                )  # task n_steps

            # record curriculum manager task probabilities
            cm = env.agent_conductor.curriculum_manager
            p_agg_stats = cm.get_agg_stats()
            for task_key, model in self.locals["models_dict"].items():
                self.logger.record(
                    f"curriculum/{task_key}_p", p_agg_stats["epoch"][task_key]
                )
                self.logger.record(
                    f"curriculum/{task_key}_ema_success",
                    env.agent_conductor.task_stats["success"].get_task_edma(task_key),
                )

            # dump and reset
            self.logger.dump(
                self.num_timesteps
            )  # very dodgy, but curreently using num_timesteps of whatever model is assigned to the callback as anchor timestep for logging...??
            env.agent_conductor.reset_epoch_stats()
            cm.reset_epoch_stats()

        # iter counter
        self.num_timesteps_multi_run += 1

        return True


class GradientCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.

    Only suitable for single policy learning (currently)
    """

    def __init__(self, verbose=0, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        # Dump
        if self.num_timesteps % self.log_freq == 0:
            grads = self.report_grad_norm()
            for key in grads.keys():
                self.logger.record(f"grads/{key}", grads[key])
        return True

    def report_grad_norm(self):
        # may add qf1, policy, etc.
        policy = self.locals["self"].policy
        grads = {}
        grads["actor"] = self.get_grad_norm(policy.actor)
        grads["critic"] = self.get_grad_norm(policy.critic)
        grads["critic_target"] = self.get_grad_norm(policy.critic_target)
        if policy.features_extractor_class is not None:
            grads["actor_encoder"] = self.get_grad_norm(policy.actor.features_extractor)
            grads["critic_encoder"] = self.get_grad_norm(
                policy.critic.features_extractor
            )
        return grads

    def get_grad_norm(self, model):
        grad_norm = []
        for p in list(filter(lambda p: p.grad is not None, model.parameters())):
            grad_norm.append(p.grad.data.norm(2).item())
        if grad_norm:
            grad_norm = np.mean(grad_norm)
        else:
            grad_norm = 0.0
        return grad_norm


class EvalCallbackMultiTask(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        eval_env_sequenced: Optional[Union[gym.Env, VecEnv]] = None,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        seperate_policies: bool = False,
        single_task_names: List[str] = None,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.seperate_policies = seperate_policies
        self.single_task_names = single_task_names
        assert len(single_task_names) > 0

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])
            if eval_env_sequenced is not None:
                eval_env_sequenced = DummyVecEnv([lambda: eval_env_sequenced])

        self.eval_env = eval_env
        self.eval_env_sequenced = eval_env_sequenced
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        for env in [self.eval_env, self.eval_env_sequenced]:
            if env is not None:
                if not isinstance(self.training_env, type(env)):
                    warnings.warn(
                        "Training and eval env are not of the same type"
                        f"{self.training_env} != {env}"
                    )

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(
        self, locals_: Dict[str, Any], globals_: Dict[str, Any]
    ) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            for env in [self.eval_env, self.eval_env_sequenced]:
                # Sync training and eval env if there is VecNormalize
                if self.model.get_vec_normalize_env() is not None and env is not None:
                    try:
                        sync_envs_normalization(self.training_env, env)
                    except AttributeError as e:
                        raise AssertionError(
                            "Training and eval env are not wrapped the same way, "
                            "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                            "and warning above."
                        ) from e
                    assert (
                        len(env.envs) == 1
                    ), "Not checked if can handle more than 1 env"

            # Test on each task individually
            for task in self.single_task_names:
                # set env eval task
                self.eval_env.envs[0].agent_conductor.set_single_task_names(
                    [task]
                )  # very hacky, but this ensures the task is always chosen
                # Reset success rate buffer
                self._is_success_buffer = []
                # set model
                if self.seperate_policies:
                    model = self.locals["models_dict"][task]
                else:
                    model = self.model
                # collect episodes
                episode_rewards, episode_lengths = evaluate_policy(
                    model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=self.render,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=self.warn,
                    callback=self._log_success_callback,
                )
                # calc stats
                mean_reward, _ = np.mean(episode_rewards), np.std(episode_rewards)
                mean_ep_length, _ = np.mean(episode_lengths), np.std(episode_lengths)
                # Add to current Logger
                self.logger.record(f"eval/{task}_mean_reward", float(mean_reward))
                self.logger.record(f"eval/{task}_mean_ep_length", mean_ep_length)
                if len(self._is_success_buffer) > 0:
                    success_rate = np.mean(self._is_success_buffer)
                    # if self.verbose >= 1:
                    #     print(f"Success rate: {100 * success_rate:.2f}%")
                    self.logger.record(
                        f"eval_success/{task}_success_rate", success_rate
                    )

            if self.eval_env_sequenced is not None:
                # Reset success rate buffer
                self._is_success_buffer = []
                # collect episodes
                episode_rewards, episode_lengths = evaluate_sequenced_policy(
                    self.locals["models_dict"],
                    self.eval_env_sequenced,
                    n_eval_episodes=self.n_eval_episodes,
                    render=self.render,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=self.warn,
                    callback=self._log_success_callback,
                )
                # calc stats
                mean_reward, _ = np.mean(episode_rewards), np.std(episode_rewards)
                mean_ep_length, _ = np.mean(episode_lengths), np.std(episode_lengths)
                # Add to current Logger
                self.logger.record("eval/sequenced_mean_reward", float(mean_reward))
                self.logger.record("eval/sequenced_mean_ep_length", mean_ep_length)
                if len(self._is_success_buffer) > 0:
                    success_rate = np.mean(self._is_success_buffer)
                    # if self.verbose >= 1:
                    #     print(f"Success rate: {100 * success_rate:.2f}%")
                    self.logger.record(
                        "eval_success/sequenced_success_rate", success_rate
                    )

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)
