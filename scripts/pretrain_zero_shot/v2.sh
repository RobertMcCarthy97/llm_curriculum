#!/bin/bash

cd ../..

python llm_curriculum/learning/train_multitask_separate.py --config llm_curriculum/learning/config/zero_shot_pretrain/v2-on_drawer_to_at_target.py

python llm_curriculum/learning/train_multitask_separate.py --config llm_curriculum/learning/config/zero_shot_pretrain/v2-on_table_to_in_drawer.py