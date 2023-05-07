import numpy as np

from llm_curriculum_algo.env_wrappers import make_env
from llm_curriculum_algo.tasks import valid_tasks


'''
TODO:
- check sequences
- check stats recording correctly
- check replay buffer
'''

def test_single_task_rollout(env, single_task_name, task_i, use_oracle_action=False, parent=None, prev_task=None):
    '''
    Checks scenrio when goal shouldn't change during episode
    '''
    
    obs = env.reset()
    
    if single_task_name is None or task_i is None:
        single_task_name = env.agent_conductor.active_task.name
        task_i = valid_tasks.index(single_task_name)
    
    # check goal correctly set
    init_one_hot_goal = obs['desired_goal']
    assert init_one_hot_goal.shape == (len(valid_tasks),)
    assert init_one_hot_goal[task_i] == 1
    assert init_one_hot_goal.sum() == 1
    
    # check prev task oracle env reset correct 9env set to correct state)
    if prev_task is not None:
        prev_task_obj = env.agent_conductor.get_task_from_name(prev_task)
        assert prev_task_obj.complete
        prev_task_success, _ = prev_task_obj.check_success_reward(obs['observation'])
        assert prev_task_success == True
        print('prev_task checks complete')
    
    for i in range(50):
        if use_oracle_action:
            action = env.get_oracle_action(obs['observation'])
        else:
            action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        # check shapes
        assert obs['observation'].shape == (28,)
        assert action.shape == (4,)
        
        # check goals
        one_hot_goal = obs['desired_goal']
        assert np.equal(one_hot_goal, init_one_hot_goal).all()
        assert info['active_task_name'] == single_task_name
        
        # check infos
        assert "success" in info
        assert info['goal_changed'] == False
        
        # check rewards
        assert (reward == 0) or (reward == -1)
        if info["success"]:
            assert reward == 0
        else:
            (reward == -1)
            
        # check parent
        if parent is not None:
            assert 'obs_parent_goal_reward' in info.keys()
            assert 'obs_parent_goal' in info.keys()
            assert info['obs_parent_goal_reward'] == reward
            assert info['obs_parent_goal'].argmax() == valid_tasks.index(parent)
        
        # check dones
        if i < 49:
            assert done == False
        else:
            assert done == True
       
    # check oracle actions succesful
    if use_oracle_action:
        assert info['success'] == True
        assert reward == 0
            
    print("test complete")

def check_multi_single_resets(env, tasks, i):
    '''
    Checks when there can be multiple tasks, but tasks houldn't change during episode
    '''
    task_counters = {task: 0 for task in tasks}
    
    for i in range(50):
        obs = env.reset()
        
        # get task info
        init_one_hot_goal = obs['desired_goal']
        init_active_task_name = env.agent_conductor.active_task.name
        init_i = init_one_hot_goal.argmax()
        
        assert init_one_hot_goal.sum() == 1
        assert valid_tasks[init_i] == init_active_task_name
        assert init_active_task_name in tasks
    
        obs, reward, done, info = env.step(env.action_space.sample())
        
        # check task not changed
        step_task_name = info['active_task_name']
        assert step_task_name in tasks
        assert step_task_name == init_active_task_name
        assert np.equal(obs['desired_goal'], init_one_hot_goal).all()
        
        task_counters[init_active_task_name] += 1
        
    # check all task occured at least once
    for task in tasks:
        assert task_counters[task] > 0
        print(f"task {task} was reset {task_counters[task]} times")
        
    # check full rollouts work
    for i in range(2):
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space.sample())
        task_name = info['active_task_name']
        task_i = valid_tasks.index(task_name)
        test_single_task_rollout(env, single_task_name=None, task_i=None)
    

def test_env():
    # TODO: check initial state
    # check parent rewards
    
    # check 1 single task
    for i, task in enumerate(valid_tasks):
        print("testing single task rollout, task: ", task, " i: ", i)
        env = make_env(
            manual_decompose_p=1, dense_rew_lowest=False, use_language_goals=False, render_mode=None, max_ep_len=50,
            contained_sequence=False, state_obs_only=False, mtenv_wrapper=False, mtenv_task_idx=None,
            single_task_names=[task],
            high_level_task_names=['move_cube_to_target'],
            )
        test_single_task_rollout(env, task, i)
        test_single_task_rollout(env, task, i, use_oracle_action=True)
    
    # check multiple single tasks
    for i in range(len(valid_tasks) - 3):
        tasks = valid_tasks[i:i+3]
        print("testing multiple single task rollout, tasks: ", tasks, " i: ", i)
        env = make_env(
            manual_decompose_p=1, dense_rew_lowest=False, use_language_goals=False, render_mode=None, max_ep_len=50,
            contained_sequence=False, state_obs_only=False, mtenv_wrapper=False, mtenv_task_idx=None,
            single_task_names=tasks,
            high_level_task_names=['move_cube_to_target'],
            )
        check_multi_single_resets(env, tasks, i)
        
    # check resets
    # TODO: check all possible combinations?...
    pairs = [
        ('move_gripper_to_cube', "grasp_cube"),
        ("cube_between_grippers", "close_gripper_cube"),
        ('pick_up_cube', "place_cube_at_target"),
    ]
    for pair in pairs:
        prev_task = pair[0]
        task = pair[1]
        print("testing resets, prev_task: ", prev_task, " task: ", task)
        env = make_env(
            manual_decompose_p=1, dense_rew_lowest=False, use_language_goals=False, render_mode=None, max_ep_len=50,
            contained_sequence=False, state_obs_only=False, mtenv_wrapper=False, mtenv_task_idx=None,
            single_task_names=[task],
            high_level_task_names=['move_cube_to_target'],
            )
        task_i = valid_tasks.index(task)
        test_single_task_rollout(env, task, task_i, prev_task=prev_task)
        
    # check parent-child pairs (only test when child and parent end at same condition)
    pairs = [
        ('move_cube_to_target', 'place_cube_at_target'),
        ('grasp_cube', 'close_gripper_cube'),
        ('pick_up_cube', 'lift_cube'),
    ]
    for pair in pairs:
        parent = pair[0]
        child = pair[1]
        print("testing parent-child pair, parent: ", parent, " child: ", child)
        env = make_env(
            manual_decompose_p=1, dense_rew_lowest=False, use_language_goals=False, render_mode=None, max_ep_len=50,
            contained_sequence=False, state_obs_only=False, mtenv_wrapper=False, mtenv_task_idx=None,
            single_task_names=[child],
            high_level_task_names=['move_cube_to_target'],
            )
        task_i = valid_tasks.index(child)
        test_single_task_rollout(env, child, task_i, parent=parent)
        

if __name__ == "__main__":
    test_env()