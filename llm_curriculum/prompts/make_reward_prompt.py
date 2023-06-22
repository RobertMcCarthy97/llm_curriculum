def make_rules() -> str:
    rules = (
        "Ensure the reward function will allow the robot to learn the task.",
        "Reply only with a reward function that returns a reward based on the current state.",
        "Do not define any additional functions.",
    )
    return "\n".join(rules)


def make_reward_function() -> str:
    """Return a skeleton reward function"""
    reward_function = (
        "def reward_function(state: Dict[str, Any]) -> float:",
        '    """Return a reward based on the current state """',
        "    reward = 0",
        "    # TODO: Add reward function here",
        "    return reward",
    )
    return "\n".join(reward_function)


def load_environment() -> str:
    with open("llm_curriculum/prompts/txt/fetch_pick_and_place_v2_desc.txt", "r") as f:
        environment = f.read()
    return environment


def load_prompt_skeleton() -> str:
    with open("llm_curriculum/prompts/txt/reward_prompt_skeleton.txt", "r") as f:
        prompt_skeleton = f.read()
    return prompt_skeleton


if __name__ == "__main__":
    prompt_skeleton = (
        load_prompt_skeleton()
        .replace("[environment]", load_environment())
        .replace("[rules]", make_rules())
        .replace("[task]", "Gripper is opened.")
        .replace("[reward function]", make_reward_function())
    )
    with open("llm_curriculum/prompts/txt/reward_prompt.txt", "w") as f:
        f.write(prompt_skeleton)
