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
    "info_keywords": ("is_success", "overall_task_success"),
    "dense_rew_tasks": [],
    "child_p_strat": None,
    "decompose_p_clip": None,
    "single_task_names": [],
    # Important params
    "wandb_name": "cube_in_drawer_to_cube_on_drawer_adapt-v3_1",
    "manual_decompose_p": 1,
    "drawer_env": True,
    "is_closed_on_reset": False,
    "cube_pos_on_reset": "in_drawer",
    "high_level_task_names": ["cube_in_drawer_to_cube_on_drawer_adapt"],
}

pretrained_models = [
    # Cube on table -> Cube on drawer
    {
        "log_path": "./models/10_July_runs/v3-cube_on_table_to_cube_on_drawer-pretrain-v3",
        "high_level_task_name": "cube_on_table_to_cube_on_drawer",
        "wandb_name": "cube_on_table_to_cube_on_drawer-pretrain-v3",
        "model_tag": "_best",
        "tasks_to_use": ["place_cube_drawer_top"],
    },
    # Cube in drawer -> cube at target
    {
        "log_path": "./models/10_July_runs/v3-in_drawer_to_at_target-pretrain-v3",
        "high_level_task_name": "cube_in_drawer_to_cube_at_target",
        "wandb_name": "in_drawer_to_at_target-pretrain-v3",
        "model_tag": "_best",
        "tasks_to_use": ["pick_up_cube"],
    },
]
