import gymnasium as gym

from llm_curriculum.envs.agent_conductor import AgentConductor
from llm_curriculum.envs.wrappers import (
    AddExtraObjectsToObsWrapper,
    CurriculumEnvWrapper,
    NonGoalNonDictObsWrapper,
    OldGymAPIWrapper,
    MTEnvWrapper,
)
from llm_curriculum.envs.curriculum_manager import (
    SeperateEpisodesCM,
)
from typing import Optional


def make_env(
    drawer_env=False,
    manual_decompose_p=1,
    dense_rew_lowest=False,
    dense_rew_tasks=[],
    use_language_goals=False,
    render_mode=None,
    max_ep_len=50,
    single_task_names=[],
    high_level_task_names=None,
    contained_sequence=False,
    state_obs_only=False,
    mtenv_wrapper=False,
    mtenv_task_idx=None,
    curriculum_manager_cls=None,
):

    if drawer_env:
        env = gym.make(
            "FetchPickAndPlaceDrawer-v2",
            render_mode=render_mode,
            is_closed_on_reset=False,  # Default: True
        )
    else:
        env = gym.make("FetchPickAndPlace-v2", render_mode=render_mode)

    env = AddExtraObjectsToObsWrapper(env, add_target=True, add_drawer=drawer_env)

    agent_conductor = AgentConductor(
        env,
        manual_decompose_p=manual_decompose_p,
        dense_rew_lowest=dense_rew_lowest,
        dense_rew_tasks=dense_rew_tasks,
        single_task_names=single_task_names,
        high_level_task_names=high_level_task_names,
        contained_sequence=contained_sequence,
        use_language_goals=use_language_goals,
    )
    env = CurriculumEnvWrapper(
        env,
        agent_conductor,
        use_language_goals=use_language_goals,
        max_ep_len=max_ep_len,
        drawer_env=drawer_env,
    )
    if curriculum_manager_cls is not None:
        curriculum_manager = curriculum_manager_cls(
            tasks_list=agent_conductor.get_possible_task_names(),
            agent_conductor=agent_conductor,
        )
        agent_conductor.set_curriculum_manager(
            curriculum_manager
        )  # TODO: not good stuff

    env = OldGymAPIWrapper(env)
    if state_obs_only:
        env = NonGoalNonDictObsWrapper(env)
    if mtenv_wrapper:
        env = MTEnvWrapper(env, mtenv_task_idx)
    return env


def make_env_baseline(name="FetchPickAndPlace-v2", render_mode=None, max_ep_len=50):
    env = gym.make(name, render_mode=render_mode)
    env = OldGymAPIWrapper(env)
    return env


def make_single_task_env(
    task_name: str = "grasp_cube", render_mode: Optional[str] = None
):
    """
    Make a Fetch environment with a single task.
    """
    return make_env(
        manual_decompose_p=1,
        dense_rew_lowest=False,
        dense_rew_tasks=[],
        use_language_goals=False,
        render_mode=render_mode,
        single_task_names=[task_name],
        high_level_task_names=["move_cube_to_target"],
        contained_sequence=False,
        curriculum_manager_cls=SeperateEpisodesCM,
    )
