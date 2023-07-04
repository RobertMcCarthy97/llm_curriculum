"""
TODO:
- Waiting more steps after success before switching policies may help
- Record success rates (implement success checkers via the high-level task)
"""

import os
import numpy as np
from copy import deepcopy
from llm_curriculum.learning.train_multitask_separate import (
    create_env,
    create_models,
    setup_logging,
)
from llm_curriculum.learning.sb3.sequenced_rollouts import evaluate_sequenced_policy
from stable_baselines3 import TD3
import wandb


def wandb_download(wandb_path, log_dir, name):
    run = wandb.init()
    artifact = run.use_artifact(wandb_path + name + ":latest")
    artifact_path = artifact.download(root=log_dir)
    return artifact_path


class ZeroShotNormedPolicy:
    def __init__(self, model, vec_env):
        self.model = model
        self.set_obs_stats_from_env(vec_env)

    def set_obs_stats_from_env(self, vec_env):
        assert not isinstance(vec_env.obs_rms, dict)
        self.obs_rms = vec_env.obs_rms
        self.epsilon = vec_env.epsilon
        self.clip_obs = vec_env.clip_obs

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


# hparams / choose task tree
hparams = {
    "initial_state_curriculum_p": 0,
    "use_baseline_env": False,
    "render_mode": "human",
    "max_ep_len": 50,
    "dense_rew_lowest": False,
    "use_language_goals": False,
    "contained_sequence": False,
    "curriculum_manager_cls": None,
    "incremental_reward": None,
    "info_keywords": ("is_success",),
    "dense_rew_tasks": [],
    "manual_decompose_p": 0,
    "drawer_env": True,
    "is_closed_on_reset": False,
    "cube_pos_on_reset": "table",
    "single_task_names": [],
    "high_level_task_names": ["move_cube_to_target"],
    "download_only": False,
}

pretrained_models = [
    ## Open drawer
    # {
    #     "log_path": "./models/test-wandb-model_save-open_drawer",
    #     "high_level_task_name": "open_drawer",
    #     "wandb_path": "robertmccarthy11/llm-curriculum/test-wandb-model_save-open_drawer_",
    # },
    # {
    #     "log_path": "./models/open_drawer-pretrained",
    #     "high_level_task_name": "open_drawer",
    # },
    # ## Close drawer
    # {
    #     "log_path": "./models/close_drawer-pretrained",
    #     "high_level_task_name": "close_drawer",
    # },
    ## Cube on table -> Cube at target
    {
        "log_path": "./models/move_cube_to_target-single_tree",
        "high_level_task_name": "move_cube_to_target",
        "wandb_path": "robertmccarthy11/llm-curriculum/move_cube_to_target-single_tree_",
    },
    # {
    #     "log_path": "./models/move_cube_to_target-pretrained",
    #     "high_level_task_name": "move_cube_to_target",
    # },
    ## Cube on drawer -> Cube in drawer
    ## Other
    # {
    #     "log_path": "./models/pick_up_cube-opened_drawer-pretrained",
    #     "high_level_task_name": "pick_up_cube",
    # },
]

# create envs
venv = create_env(
    hparams,
    eval=True,
).venv

# create models
models_dict = {}

for pretrained in pretrained_models:
    pretrained_params = deepcopy(hparams)
    pretrained_params["high_level_task_names"] = [pretrained["high_level_task_name"]]
    log_path = pretrained["log_path"]

    # create envs - for obs norm stats
    if "wandb_path" in pretrained.keys():
        vec_norm_path = wandb_download(
            pretrained["wandb_path"], log_path, "vec_norm_env"
        )
        vec_norm_path = os.path.join(vec_norm_path, "vec_norm_env.pkl")
    else:
        vec_norm_path = os.path.join(log_path, "vec_norm_env.pkl")
    vec_norm_env = create_env(
        pretrained_params,
        eval=True,
        vec_norm_path=vec_norm_path,
    )

    # load pretrained policies
    possible_tasks = vec_norm_env.envs[0].agent_conductor.get_possible_task_names()
    for task_name in possible_tasks:
        if "wandb_path" in pretrained.keys():
            model_path = wandb_download(pretrained["wandb_path"], log_path, task_name)
            model_path = os.path.join(model_path, task_name)
        else:
            model_path = os.path.join(log_path, "models", task_name)
        # model = TD3("MlpPolicy", env)
        model = TD3.load(model_path, env=venv)
        model = ZeroShotNormedPolicy(model, vec_norm_env)
        models_dict[task_name] = model

# perform evaluation
if not hparams["download_only"]:
    episode_rewards, episode_lengths = evaluate_sequenced_policy(
        models_dict,
        venv,
        n_eval_episodes=10,
        render=True,
        deterministic=True,
        return_episode_rewards=True,
        warn=True,
        callback=None,
        verbose=1,
    )
    print("episode_rewards:", episode_rewards)
    print("episode_lengths:", episode_lengths)
