"""
TODO:
VecNormEnvs are killing me:
- Need to save norm stats for each model
- Create non-normed env
- Load norm stats for each model
- Create model wrapper that loads norm stats and norms obs before passing to model
"""


import os

from llm_curriculum.learning.train_multitask_separate import (
    create_env,
    create_models,
    setup_logging,
)
from llm_curriculum.learning.sb3.sequenced_rollouts import evaluate_sequenced_policy

from stable_baselines3 import TD3

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
    "high_level_task_names": ["close_drawer"],
    "contained_sequence": False,
    "curriculum_manager_cls": None,
    "incremental_reward": None,
    "info_keywords": ("is_success",),
    "log_path": "./logs/" + "26_06_2023-16_15_33",
}


# create envs
env = create_env(
    hparams,
    eval=True,
    vec_norm_path=os.path.join(hparams["log_path"], "vec_norm_env.pkl"),
)

# # setup logging
# logger, callback = setup_logging(hparams, env)
# assert len(env.envs) == 1
# env.envs[0].agent_conductor.set_logger(logger)

# create models
models_dict = {}
# logger = None
# models_dict = create_models(env, logger, hparams)
possible_tasks = env.envs[0].agent_conductor.get_possible_task_names()

# load pretrained policies
for task_name in possible_tasks:
    # model = TD3("MlpPolicy", env)
    model = TD3.load(os.path.join(hparams["log_path"], "models", task_name), env=env)
    models_dict[task_name] = model

# perform evaluation
episode_rewards, episode_lengths = evaluate_sequenced_policy(
    models_dict,
    env,
    n_eval_episodes=10,
    render=True,
    deterministic=True,
    return_episode_rewards=True,
    warn=True,
    callback=None,
)
print("episode_rewards:", episode_rewards)
print("episode_lengths:", episode_lengths)
