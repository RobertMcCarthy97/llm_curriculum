#!/bin/bash

cd ../..

python llm_curriculum/learning/train_multitask_separate.py --config llm_curriculum/learning/config/zero_shot_pretrain/v3-on_table_to_on_drawer.py

python llm_curriculum/learning/train_multitask_separate.py --config llm_curriculum/learning/config/zero_shot_pretrain/v3-in_drawer_to_at_target.py