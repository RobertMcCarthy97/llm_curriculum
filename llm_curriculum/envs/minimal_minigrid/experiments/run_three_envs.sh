python llm_curriculum/envs/minimal_minigrid/train.py \
    --algo ppo \
    --env MiniGrid-IsNextTo-6x6-N2-v0 \
    --conf-file llm_curriculum/envs/minigrid/hyperparams/ppo.yml \
    --wandb-group-name minimal_minigrid \
    --track

python llm_curriculum/envs/minimal_minigrid/train.py \
    --algo ppo \
    --env MiniGrid-IsNextTo-6x6-N2-DecomposedReward-v0 \
    --conf-file llm_curriculum/envs/minigrid/hyperparams/ppo.yml \
    --wandb-group-name minimal_minigrid \
    --track

python llm_curriculum/envs/minimal_minigrid/train.py \
    --algo ppo \
    --env MiniGrid-UnlockRed-v0 \
    --conf-file llm_curriculum/envs/minigrid/hyperparams/ppo.yml \
    --wandb-group-name minimal_minigrid \
    --track

python llm_curriculum/envs/minimal_minigrid/train.py \
    --algo ppo \
    --env MiniGrid-UnlockRed-DecomposedReward-v0 \
    --conf-file llm_curriculum/envs/minigrid/hyperparams/ppo.yml \
    --wandb-group-name minimal_minigrid \
    --track