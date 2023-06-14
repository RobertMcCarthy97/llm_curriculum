#!/bin/bash

# Run baseline experiments
for env_id in grasp_cube lift_cube pick_up_cube
do
    python llm_curriculum/learning/sb3_td3.py \
        --env-id $env_id \
        --exp-name sb3_td3 \
        --wandb-entity ucl-air-lab \
        --wandb-project-name llm_curriculum \
        --wandb-group-name value_bootstrap \
        --track \
        --total-timesteps 1000000 \
        --capture-video \
        --video-frequency 100000 \
        --eval-frequency 10000 \
        --checkpoint-frequency 10000
done

# TODO: Add bootstrap experiment