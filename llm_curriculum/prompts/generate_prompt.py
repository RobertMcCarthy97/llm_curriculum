def make_rules() -> str:
    rules = (
        "For each element <goal>, think of child elements consisting of intermediate landmark states which would be beneficial to reaching the parent goal state.",
        "Always start the description with [start of description] and end it with [end of description].",
        "You can assume that the robot is capable of doing anything, even for the most challenging task.",
        "As far as possible, aim to re-use the same descriptions when adding child elements to the tree.",
        "Keep adding children until each leaf node in the task tree is atomic and cannot be decomposed further.",
        "All elements in the tree should be <goal> elements with the 'name' attribute specified.",
    )
    return "\n".join(rules)


def make_plan() -> str:
    plan = (
        "[start of description]",
        "<goal name='Cube is at the target' />",
        "[end of description]",
    )
    return "\n".join(plan)


def load_environment() -> str:
    with open("llm_curriculum/prompts/txt/fetch_pick_and_place_v2_desc.txt", "r") as f:
        environment = f.read()
    return environment


def load_prompt_skeleton() -> str:
    with open("llm_curriculum/prompts/txt/prompt_skeleton.txt", "r") as f:
        prompt_skeleton = f.read()
    return prompt_skeleton


if __name__ == "__main__":
    prompt_skeleton = (
        load_prompt_skeleton()
        .replace("[environment]", load_environment())
        .replace("[rules]", make_rules())
        .replace("[plan]", make_plan())
    )
    with open("llm_curriculum/prompts/txt/prompt.txt", "w") as f:
        f.write(prompt_skeleton)
