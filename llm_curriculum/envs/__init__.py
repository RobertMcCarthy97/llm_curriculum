import gymnasium as gym

from llm_curriculum.envs.agent_conductor import AgentConductor
from llm_curriculum.envs.wrappers import (
    AddTargetToObsWrapper,
    CurriculumEnvWrapper,
    NonGoalNonDictObsWrapper,
    OldGymAPIWrapper,
    MTEnvWrapper,
)


def make_env(
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

    env = gym.make("FetchPickAndPlace-v2", render_mode=render_mode)
    env = AddTargetToObsWrapper(env)

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