# hparams / choose task tree
hparams = {
    "initial_state_curriculum_p": 0,
    "use_baseline_env": False,
    "render_mode": "human",
    "max_ep_len": 100,
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
    "wandb_name": "open_then_in_drawer_then_close-0_shot adapt",
    "manual_decompose_p": 1,
    "task_complete_thresh": 6,
    "drawer_env": True,
    "is_closed_on_reset": True,
    "cube_pos_on_reset": "table",
    "high_level_task_names": ["open_drawer_then_cube_in_drawer_then_close_adapt"],
}

pretrained_models = [
    # Open drawer
    {
        "log_path": "./models/10_July_runs/open_drawer-pretrained",
        "high_level_task_name": "open_drawer",
        "wandb_name": "unknown",
        "model_tag": "_best",
        "tasks_to_use": ["open_drawer"],
    },
    # On table -> In drawer
    {
        "log_path": "./models/10_July_runs/v2-on_table_to_in_drawer-pretrain-v2",
        "high_level_task_name": "cube_on_table_to_cube_in_drawer",
        "wandb_name": "on_table_to_in_drawer-pretrain-v2",
        "model_tag": "_best",
        "tasks_to_use": ["place_cube_drawer"],
    },
    # Close drawer
    {
        "log_path": "./models/10_July_runs/close_drawer-pretrained",
        "high_level_task_name": "close_drawer",
        "wandb_name": "close_drawer-pretrained",
        "model_tag": "_best",
        "tasks_to_use": ["close_drawer"],
    },
]
