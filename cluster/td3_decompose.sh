#!/bin/bash -l

#$ -l h_rt=48:00:00      # Request 48 hours runtime
#$ -l h_vmem=8G          # Request 8GB of memory
#$ -pe smp 16            # Request 16 CPU cores
#$ -wd /home/ucabdc6/github/llm_curriculum

module load python/3.9.10
module load default-modules

.venv/bin/python llm_curriculum/learning/train_multitask_separate.py \
    --config llm_curriculum/learning/config/default.py \
    --config.wandb.name='td3_decompose' \
    --config.wandb.track=True