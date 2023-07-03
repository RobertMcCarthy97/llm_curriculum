python llm_curriculum/envs/minigrid/train.py \
    --algo ppo \
    --env MiniGrid-UnlockDecomposedAutomated-v0 \
    --conf-file llm_curriculum/envs/minigrid/hyperparams/ppo.yml \
    --wandb-group-name minigrid_2 \
    --track

python llm_curriculum/envs/minigrid/train.py \
    --algo ppo \
    --env MiniGrid-UnlockPickupDecomposedAutomated-v0 \
    --conf-file llm_curriculum/envs/minigrid/hyperparams/ppo.yml \
    --wandb-group-name minigrid_2 \
    --track

python llm_curriculum/envs/minigrid/train.py \
    --algo ppo \
    --env MiniGrid-BlockedUnlockPickupDecomposedAutomated-v0 \
    --conf-file llm_curriculum/envs/minigrid/hyperparams/ppo.yml \
    --wandb-group-name minigrid_2 \
    --track