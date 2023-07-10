# hparams / choose task tree
hparams = {
    "initial_state_curriculum_p": 0,
    "use_baseline_env": False,
    "render_mode": "human",
    "max_ep_len": 50,
    "dense_rew_lowest": False,
    "use_language_goals": False,
    "contained_sequence": False,
    "curriculum_manager_cls": None,
    "incremental_reward": None,
    "info_keywords": ("is_success",),
    "dense_rew_tasks": [],
    "child_p_strat": None,
    "decompose_p_clip": None,
    # Important params
    "manual_decompose_p": 1,
    "drawer_env": False,
    "is_closed_on_reset": False,
    "cube_pos_on_reset": "table",
    "single_task_names": [],
    "high_level_task_names": ["pick_up_cube"],
    "download_only": False,
}

pretrained_models = [
    ## Open drawer
    # {
    #     "log_path": "./models/test-wandb-model_save-open_drawer",
    #     "high_level_task_name": "open_drawer",
    #     "wandb_path": "robertmccarthy11/llm-curriculum/test-wandb-model_save-open_drawer_",
    # },
    # {
    #     "log_path": "./models/open_drawer-pretrained",
    #     "high_level_task_name": "open_drawer",
    # },
    # ## Close drawer
    # {
    #     "log_path": "./models/close_drawer-pretrained",
    #     "high_level_task_name": "close_drawer",
    # },
    ## Cube on table -> Cube at target
    # {
    #     "log_path": "./models/move_cube_to_target-single_tree",
    #     "high_level_task_name": "move_cube_to_target",
    #     "wandb_path": "robertmccarthy11/llm-curriculum/move_cube_to_target-single_tree_",
    # },
    # {
    #     "log_path": "./models/move_cube_to_target-pretrained",
    #     "high_level_task_name": "move_cube_to_target",
    # },
    ## Cube on drawer -> Cube in drawer
    {
        "log_path": "./models/test-pick_up_cube",
        "high_level_task_name": "pick_up_cube",
        "wandb_name": "test-pick_up_cube",
        "model_tag": "_best",
    },
]
