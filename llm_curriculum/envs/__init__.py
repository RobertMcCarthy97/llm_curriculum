from gymnasium.envs.registration import register
from llm_curriculum.envs.make_env import (
    make_env,  # noqa: F401
    make_env_baseline,  # noqa: F401
    make_single_task_env,
)

# Register environments
register(
    id="FetchGraspCube-v2",
    entry_point=make_single_task_env,
    kwargs={
        "task_name": "grasp_cube",
    },
)

register(
    id="FetchLiftCube-v2",
    entry_point=make_single_task_env,
    kwargs={
        "task_name": "lift_cube",
    },
)
