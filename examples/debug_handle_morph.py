from llm_curriculum.envs.make_env import make_env
from llm_curriculum.envs.curriculum_manager import (
    SeperateEpisodesCM,
)
from llm_curriculum.envs.cli import get_user_action

import numpy as np

if __name__ == "__main__":

    env = make_env(
        drawer_env=True,
        manual_decompose_p=1,
        dense_rew_lowest=False,
        dense_rew_tasks=[],
        use_language_goals=False,
        render_mode="human",
        single_task_names=["pull_handle_to_open"],
        high_level_task_names=["open_drawer"],
        contained_sequence=False,
        curriculum_manager_cls=None,
        use_incremental_reward=True,
        is_closed_on_reset=True,
        cube_pos_on_reset="table",
    )

    obs = env.reset()
    print("env reset")

    for _ in range(35):
        ## Actions
        # action = get_user_action()
        action = np.array([0, -1, 0, 0])

        # step
        input("Press [Enter] to step environment.")
        obs, reward, done, info = env.step(action)
        env.render()
