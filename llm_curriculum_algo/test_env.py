from llm_curriculum_algo.env_wrappers import make_env


def test_env():
    
    env = make_env(manual_decompose_p=1, dense_rew_lowest=False, use_language_goals=False, render_mode=None, max_ep_len=50, single_task_names=['pick_up_cube', 'lift_cube'], high_level_task_names=['move_cube_to_target'], contained_sequence=False, state_obs_only=False, mtenv_wrapper=False, mtenv_task_idx=None)
    
    obs = env.reset()
    
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        assert obs['observation'].shape == (28,)
        assert action.shape == (4,)
        assert "success" in info
        assert (reward == 0) or (reward == -1)
        if info["success"]:
            assert reward == 0
        else:
            (reward == -1)
            
    print("test complete")
    

if __name__ == "__main__":
    test_env()