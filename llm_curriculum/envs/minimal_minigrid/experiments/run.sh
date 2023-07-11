WANDB_GROUP=${1:-minimal_minigrid}
ENV_IDS=(
    "MiniGrid-IsNextTo-6x6-v0"
    "MiniGrid-IsNextTo-6x6-DecomposedReward-v0"
    "MiniGrid-IsNextTo-6x6-DecomposedReward-NoMission-v0"
    "MiniGrid-IsNextTo-6x6-DecomposedReward-NoReward-v0"
    "MiniGrid-IsNextTo-6x6-DecomposedReward-NoMission-NoReward-v0"
    # "MiniGrid-IsNextTo-12x12-v0"
    # "MiniGrid-IsNextTo-12x12-DecomposedReward-v0"
    # "MiniGrid-UnlockRed-6x6-v0"
    # "MiniGrid-UnlockRed-6x6-DecomposedReward-v0"
    # "MiniGrid-UnlockRed-12x12-v0"
    # "MiniGrid-UnlockRed-12x12-DecomposedReward-v0"
)

echo "Running experiments" 
echo "ENV_IDS: ${ENV_IDS[@]}"
echo "WANDB_GROUP: $WANDB_GROUP"

for env_id in ${ENV_IDS[@]}
do
    python llm_curriculum/envs/minimal_minigrid/train.py \
        --algo ppo \
        --env $env_id \
        --conf-file llm_curriculum/envs/minimal_minigrid/hyperparams/ppo.yml \
        --wandb-group-name $WANDB_GROUP
        # --track
done