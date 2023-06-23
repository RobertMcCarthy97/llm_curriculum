import pytest

from llm_curriculum.envs import make_env, make_env_baseline, make_single_task_env
from llm_curriculum.envs.curriculum_manager import SeperateEpisodesCM


def test_make_env():
    env = make_env(
        drawer_env=False,
        use_incremental_reward=False,
        manual_decompose_p=1,
        dense_rew_lowest=False,
        dense_rew_tasks=[],
        use_language_goals=False,
        render_mode="rgb_array",
        single_task_names=["lift_cube", "pick_up_cube"],
        high_level_task_names=["move_cube_to_target"],
        contained_sequence=False,
        curriculum_manager_cls=SeperateEpisodesCM,
    )

    env.reset()


def test_make_env_baseline():
    env = make_env_baseline()
    env.reset()


env_ids = ["grasp_cube", "lift_cube", "pick_up_cube"]


@pytest.mark.parametrize("env_id", env_ids)
def test_registered_environments(env_id):
    # Initialize the environment
    env = make_single_task_env(env_id)

    # Reset the environment
    obs = env.reset()

    assert isinstance(obs, dict)  # Check if observation is a dictionary
    assert (
        "observation" in obs
    )  # Check if "observation" key is present in the observation

    done = False
    while not done:
        action = env.action_space.sample()  # Sample a random action
        obs, reward, done, info = env.step(action)

        assert isinstance(obs, dict)  # Check if observation is a dictionary
        assert (
            "observation" in obs
        )  # Check if "observation" key is present in the observation
        assert isinstance(reward, float)  # Check if reward is a float
        assert isinstance(done, bool)  # Check if done is a boolean
        assert isinstance(info, dict)  # Check if info is a dictionary
