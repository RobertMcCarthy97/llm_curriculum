from llm_curriculum.envs.wrappers import make_env
from llm_curriculum.envs.curriculum_manager import (
    SeperateEpisodesCM,
)

if __name__ == "__main__":

    env = make_env(
        manual_decompose_p=1,
        dense_rew_lowest=False,
        dense_rew_tasks=[],
        use_language_goals=False,
        render_mode="human",
        single_task_names=["lift_cube", "pick_up_cube"],
        high_level_task_names=["pick_up_cube"],
        contained_sequence=False,
        curriculum_manager_cls=SeperateEpisodesCM,
    )

    # env = make_env(
    #     manual_decompose_p=1,
    #     dense_rew_lowest=False,
    #     dense_rew_tasks=[],
    #     use_language_goals=False,
    #     render_mode="human",
    #     single_task_names=["lift_cube"],
    #     high_level_task_names=["move_cube_to_target"],
    #     contained_sequence=False,
    #     curriculum_manager_cls=SeperateEpisodesCM
    # )

    for _ in range(5):

        obs = env.reset()
        print("env reset")

        for _ in range(35):
            ## Actions
            # action = env.action_space.sample()
            # action = get_user_action()
            action = env.get_oracle_action(obs["observation"])

            # step
            # obs, reward, terminated, truncated, info = env.step(action)
            obs, reward, done, info = env.step(action)
            env.render()

            # prints
            active_task = info["active_task_name"]
            print(f"Active Task: {active_task}")
            print(f"Goal: {obs['desired_goal']}")
            # print(f"Obs: {obs['observation'].shape}")
            # print(f"step count: {env.ep_steps}")
            print(f"success: {info['is_success']}")
            print(f"Reward: {reward}")
            print("done: ", done)
            # print(f"info: {info}")
            print("Parent goal: ", info.get("obs_parent_goal", None))
            print("Parent goal reward: ", info.get("obs_parent_goal_reward", None))
            print()

            # time.sleep(0.1)

        stats = env.agent_conductor.get_stats()
        print(stats)
        env.agent_conductor.reset_epoch_stats()