from llm_curriculum.envs import make_env, make_env_baseline
from llm_curriculum.envs.curriculum_manager import SeperateEpisodesCM


def test_make_env():
    env = make_env(
        manual_decompose_p=1,
        dense_rew_lowest=False,
        dense_rew_tasks=[],
        use_language_goals=False,
        render_mode="rgb_array",
        single_task_names=["lift_cube", "pick_up_cube"],
        high_level_task_names=["pick_up_cube"],
        contained_sequence=False,
        curriculum_manager_cls=SeperateEpisodesCM,
    )

    env.reset()


def test_make_env_baseline():
    env = make_env_baseline()
    env.reset()
