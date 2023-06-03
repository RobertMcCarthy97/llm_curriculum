import gymnasium as gym 

def test_fetch_pick_and_place():
    env = gym.make('FetchPickAndPlace-v2')
    env.reset()
    for _ in range(100):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
    env.close()