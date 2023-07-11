"""
TODO:
- Waiting more steps after success before switching policies may help
"""

import os
import numpy as np
from copy import deepcopy
from typing import Any, Dict, Optional, Union, List

from llm_curriculum.learning.train_multitask_separate import create_env
from llm_curriculum.learning.sb3.sequenced_rollouts import evaluate_sequenced_policy
from llm_curriculum.learning.sb3.td3_custom import TD3
import torch as th
import gym

from PIL import Image, ImageDraw
from stable_baselines3.common.logger import Video


############################
# Settings
############################

# Choose run
from llm_curriculum.learning.config.zero_shot_adapt.v2_2 import (
    hparams,
    pretrained_models,
)

# W&B tracking
do_wandb = True

##############################
# End of settings
##############################


class VideoRecorder:
    def __init__(
        self,
        eval_env: gym.Env,
        models_dict: dict,
        n_eval_episodes: int = 1,
        add_text: bool = False,
    ):
        super().__init__()
        self._eval_env = eval_env
        self.models_dict = models_dict
        self._n_eval_episodes = n_eval_episodes
        self.add_text = add_text

    def record_video(self, deterministic=True):
        screens = []

        def overlay_text(rgb_array: np.array, info: Dict[str, Any] = None):
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

        # Rollout policy
        evaluate_sequenced_policy(
            self.models_dict,
            self._eval_env,
            callback=grab_screens,
            n_eval_episodes=self._n_eval_episodes,
            deterministic=deterministic,
        )

        # Log to W&B
        wandb.log(
            {
                # "Video": Video(th.ByteTensor(np.array([screens])), fps=25)
                f"Video": wandb.Video(np.array([screens]), fps=25, format="gif")
            }
        )


# Custom class to normalize agent observations
class ZeroShotNormedPolicy:
    def __init__(self, model):
        self.model = model
        self.set_obs_stats()

    def set_obs_stats(self):
        self.obs_rms = self.model.obs_norm_save_stats["obs_rms"]
        self.epsilon = self.model.obs_norm_save_stats["epsilon"]
        self.clip_obs = self.model.obs_norm_save_stats["clip_obs"]

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        assert not isinstance(obs, dict)
        obs = deepcopy(obs)
        normed_obs = np.clip(
            (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
            -self.clip_obs,
            self.clip_obs,
        )
        return normed_obs.astype(np.float32)

    def predict(
        self, observations, state=None, episode_start=None, deterministic=False
    ):
        # norm observations
        observations = self._normalize_obs(observations)
        # predict
        actions, states = self.model.predict(
            observations,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        # return
        return actions, states


# Create env
if do_wandb:
    hparams["render_mode"] = "rgb_array"
venv = create_env(
    hparams,
    eval=True,
).venv

# Create models
models_dict = {}

for pretrained in pretrained_models:
    pretrained_params = deepcopy(hparams)
    pretrained_params["high_level_task_names"] = [pretrained["high_level_task_name"]]

    log_path = pretrained["log_path"]
    tag = pretrained["model_tag"]
    tasks_to_use = pretrained["tasks_to_use"]

    # # create envs - for obs norm stats
    # vec_norm_path = os.path.join(log_path, "vec_norm_env.pkl")
    # vec_norm_env = create_env(
    #     pretrained_params,
    #     eval=True,
    #     vec_norm_path=vec_norm_path,
    # )
    # possible_tasks = vec_norm_env.envs[0].agent_conductor.get_possible_task_names()

    # load pretrained policies
    for task_name in tasks_to_use:
        model_path = os.path.join(log_path, "models", task_name + tag)
        model = TD3.load(model_path, env=venv)
        model = ZeroShotNormedPolicy(model)
        models_dict[task_name] = model

# Perform evaluation
(
    episode_rewards,
    episode_lengths,
    episode_successes,
    episode_high_level_task_successes,
) = evaluate_sequenced_policy(
    models_dict,
    venv,
    n_eval_episodes=20,
    render=True,
    deterministic=True,
    return_episode_rewards=True,
    warn=True,
    callback=None,
    verbose=1,
)
print("episode_rewards:", episode_rewards)
print("episode_lengths: ", episode_lengths)
print("episode_successes: ", episode_successes)
print("episode_high_level_task_successes: ", episode_high_level_task_successes)
print()
print("mean episode_rewards: ", np.mean(episode_rewards))
print("mean episode_successes: ", np.mean(episode_successes))
print(
    "mean episode_high_level_task_successes: ",
    np.mean(episode_high_level_task_successes),
)


# Log to W&B
if do_wandb:
    import wandb

    run = wandb.init(
        entity="robertmccarthy11",
        project="llm-curriculum",
        group="0-shot_adapt",
        name=hparams["wandb_name"],
        job_type="eval",
        config=hparams,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    wandb.log({"pretrained_models": pretrained_models})

    wandb.log(
        {
            "mean_episode_rewards": np.mean(episode_rewards),
            "mean_episode_successes": np.mean(episode_successes),
            "mean_episode_high_level_task_successes": np.mean(
                episode_high_level_task_successes
            ),
            "episode_rewards": episode_rewards,
            "episode_successes": episode_successes,
            "episode_high_level_task_successes": episode_high_level_task_successes,
        }
    )

    vid_recorder = VideoRecorder(
        venv,
        models_dict,
        n_eval_episodes=10,
        add_text=True,
    )
    vid_recorder.record_video(deterministic=True)

    run.finish()
