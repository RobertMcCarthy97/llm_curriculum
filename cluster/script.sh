#!/bin/bash -l

#$ -l h_rt=4:00:00      # Request 4 hours runtime
#$ -l h_vmem=8G         # Request 8GB of memory
#$ -pe smp 4            # Request 4 CPU cores
#$ -wd /home/ucabdc6/github/llm_curriculum

module load python/3.9.10
module load default-modules

.venv/bin/python llm_curriculum/learning/baselines/vanilla_rl.py \
    --config llm_curriculum/learning/baselines/config/default.py \
    --config.use_her=True \
    --config.wandb.name='td3_her' \
    --config.wandb.track=True
