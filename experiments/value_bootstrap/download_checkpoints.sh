#!/bin/bash 


python scripts/download_wandb_checkpoint.py --wandb-run-path ucl-air-lab/llm_curriculum/rkw0ak6u --wandb-filename model.zip --save-prefix grasp_cube
python scripts/download_wandb_checkpoint.py --wandb-run-path ucl-air-lab/llm_curriculum/zr09v14e --wandb-filename model.zip --save-prefix lift_cube
python scripts/download_wandb_checkpoint.py --wandb-run-path ucl-air-lab/llm_curriculum/hbkxbkq4 --wandb-filename model.zip --save-prefix pick_up_cube

for env_id in grasp_cube lift_cube pick_up_cube
do
    python llm_curriculum/learning/sb3/td3_eval.py --env-id $env_id --exp-name None --capture-video
done

# Videos saved to 'logs/XXX/videos'