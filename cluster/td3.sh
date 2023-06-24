#!/bin/bash -l

#$ -l h_rt=24:00:00      # Request 24 hours runtime
#$ -l h_vmem=8G          # Request 8GB of memory
#$ -pe smp 16            # Request 16 CPU cores
#$ -wd /home/ucabdc6/github/llm_curriculum

module load python/3.9.10
module load default-modules

.venv/bin/python llm_curriculum/learning/baselines/vanilla_rl.py \
    --config llm_curriculum/learning/config/baseline.py \
    --config.use_her=False \
    --config.wandb.name='td3' \
    --config.wandb.track=True