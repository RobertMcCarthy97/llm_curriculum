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
)
from llm_curriculum.learning.sb3.sequenced_rollouts import evaluate_sequenced_policy
from llm_curriculum.learning.sb3.td3_custom import TD3


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


# Choose run
from llm_curriculum.learning.config.zero_shot_adapt.sample_run import (
    hparams,
    pretrained_models,
)

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
    tag = pretrained["model_tag"]

    # create envs - for obs norm stats
    vec_norm_path = os.path.join(log_path, "vec_norm_env.pkl")
    vec_norm_env = create_env(
        pretrained_params,
        eval=True,
        vec_norm_path=vec_norm_path,
    )

    # load pretrained policies
    possible_tasks = vec_norm_env.envs[0].agent_conductor.get_possible_task_names()
    for task_name in possible_tasks:
        model_path = os.path.join(log_path, "models", task_name + tag)
        # model = TD3("MlpPolicy", env)
        model = TD3.load(model_path, env=venv)
        model = ZeroShotNormedPolicy(model)
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
