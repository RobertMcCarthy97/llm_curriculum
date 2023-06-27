import gymnasium as gym


def test_drawer_env():
    env = gym.make(
        "FetchPickAndPlaceDrawer-v2",
        render_mode="human",
        is_closed_on_reset=False,  # Default: True
        is_cube_inside_drawer=False,  # Default: True
    )
    obs = env.reset()

    # Use these methods to reset the drawer state

    # env.reset_drawer_open()
    env.reset_drawer_closed()

    # env.reset_cube_outside_drawer()
    env.reset_cube_inside_drawer()
