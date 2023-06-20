import argparse
from llm_curriculum.envs import make_single_task_env


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="grasp_cube")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    env = make_single_task_env(args.env, render_mode="human")

    for _ in range(5):

        obs = env.reset()
        print("env reset")

        ep_return = 0
        for _ in range(50):
            ## Actions
            # action = env.action_space.sample()
            # action = get_user_action()
            action = env.get_oracle_action(obs["observation"])

            # step
            # obs, reward, terminated, truncated, info = env.step(action)
            obs, reward, done, info = env.step(action)
            ep_return += reward
            env.render()

            # prints
            active_task = info["active_task_name"]
            if done:
                break

            # time.sleep(0.1)
        print(f"ep return: {ep_return}")

        stats = env.agent_conductor.get_stats()
        if args.verbose:
            print(stats)
        env.agent_conductor.reset_epoch_stats()
