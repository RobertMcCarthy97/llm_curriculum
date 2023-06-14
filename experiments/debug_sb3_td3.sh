#!/bin/bash

# Run debug experiment
for env_id in grasp_cube
do
    python llm_curriculum/learning/sb3_td3.py \
        --env-id $env_id \
        --exp-name sb3_td3 \
        --wandb-entity ucl-air-lab \
        --wandb-project-name llm_curriculum \
        --wandb-group-name value_bootstrap \
        --track \
        --total-timesteps 10000 \
        --capture-video \
        --video-frequency 5000 \
        --eval-frequency 1000 \
        --checkpoint-frequency 1000
done