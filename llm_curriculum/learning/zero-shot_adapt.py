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
    "drawer_env": True,
    "manual_decompose_p": 1,
    "dense_rew_lowest": False,
    "dense_rew_tasks": [],
    "use_language_goals": False,
    "single_task_names": [],
    "high_level_task_names": ["open_drawer_then_pickup_cube"],
    "contained_sequence": False,
    "curriculum_manager_cls": None,
    "incremental_reward": None,
    "info_keywords": ("is_success",),
}

pretrained_models = [
    {
        "log_path": "./models/open_drawer-pretrained",
        "high_level_task_name": "open_drawer",
    },
    {
        "log_path": "./models/pick_up_cube-opened_drawer-pretrained",
        "high_level_task_name": "pick_up_cube",
    },
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
    vec_norm_env = create_env(
        pretrained_params,
        eval=True,
        vec_norm_path=os.path.join(log_path, "vec_norm_env.pkl"),
    )
    # load pretrained policies
    possible_tasks = vec_norm_env.envs[0].agent_conductor.get_possible_task_names()
    for task_name in possible_tasks:
        # model = TD3("MlpPolicy", env)
        model = TD3.load(os.path.join(log_path, "models", task_name), env=venv)
        model = ZeroShotNormedPolicy(model, vec_norm_env)
        models_dict[task_name] = model

# perform evaluation
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
