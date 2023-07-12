from llm_curriculum.envs.make_env import make_env
from llm_curriculum.envs.curriculum_manager import (
    SeperateEpisodesCM,
)
from llm_curriculum.envs.cli import get_user_action

if __name__ == "__main__":

    env = make_env(
        drawer_env=True,
        manual_decompose_p=1,
        dense_rew_lowest=False,
        dense_rew_tasks=[],
        use_language_goals=False,
        render_mode="human",
        single_task_names=[],
        high_level_task_names=["open_then_place_drawer_low"],
        contained_sequence=False,
        curriculum_manager_cls=None,
        use_incremental_reward=False,
        task_complete_thresh=3,
        initial_state_curriculum_p=0,
        # drawer env
        is_closed_on_reset=True,
        cube_pos_on_reset="table",
    )

    # set traversal mode
    # env.agent_conductor.set_tree_traversal_mode('exploit')

    for _ in range(5):

        obs = env.reset()
        print("env reset")

        for _ in range(100):
            print("cube_pos_obs: ", obs["observation"][3:6])
            ## Actions
            # action = env.action_space.sample()
            # action = get_user_action()
            action = env.get_oracle_action(obs["observation"])

            # step
            # obs, reward, terminated, truncated, info = env.step(action)
            input()
            obs, reward, done, info = env.step(action)
            env.render()

            # prints
            active_task = info["active_task_name"]
            print(f"Active Task: {active_task}")
            # print(f"Goal: {obs['desired_goal']}")
            # print(f"Obs: {obs['observation'].shape}")
            # print(f"step count: {env.ep_steps}")
            # print(f"success: {info['is_success']}")
            print(f"Reward: {reward}")
            # print("done: ", done)
            # print(f"info: {info}")
            # print("Parent goal: ", info.get("obs_parent_goal", None))
            # print("Parent goal reward: ", info.get("obs_parent_goal_reward", None))
            print()

            # time.sleep(0.1)

        stats = env.agent_conductor.get_stats()
        print(stats)
        env.agent_conductor.reset_epoch_stats()
