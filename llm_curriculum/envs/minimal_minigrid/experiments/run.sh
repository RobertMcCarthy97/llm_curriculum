ENV_IDS=(
    "MiniGrid-IsNextTo-6x6-N2-v0"
    "MiniGrid-IsNextTo-6x6-N2-DecomposedReward-v0"
    "MiniGrid-UnlockRed-v0"
    "MiniGrid-UnlockRed-DecomposedReward-v0"
)

for env_id in ${ENV_IDS[@]}
do
    python llm_curriculum/envs/minimal_minigrid/train.py \
        --algo ppo \
        --env $env_id \
        --conf-file llm_curriculum/envs/minimal_minigrid/hyperparams/ppo.yml \
        --wandb-group-name minimal_minigrid \
        --track
done