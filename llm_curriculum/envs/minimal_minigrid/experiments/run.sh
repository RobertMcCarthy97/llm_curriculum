WANDB_GROUP=${1:-""}
WANDB_STR=$( [[ -n "$WANDB_GROUP" ]] && echo "--wandb-group-name $WANDB_GROUP --track" || echo "" )
DEBUG_STR=$( [[ -n "$WANDB_GROUP" ]] && echo "" || echo "--hyperparams n_timesteps:100 n_envs:1" )
ENV_IDS=(
    "MiniGrid-UnlockPickup-v0"
    "MiniGrid-UnlockPickup-OracleDecomposedReward-v0"
    "MiniGrid-UnlockPickup-OracleDecomposedReward-NoMission-v0"
    "MiniGrid-UnlockPickup-OracleDecomposedReward-NoReward-v0"
    "MiniGrid-UnlockPickup-OracleDecomposedReward-NoMission-NoReward-v0"
    # "MiniGrid-UnlockPickup-DecomposedReward-v0"
    # "MiniGrid-UnlockPickup-DecomposedReward-NoMission-v0"
    # "MiniGrid-UnlockPickup-DecomposedReward-NoReward-v0"
    # "MiniGrid-UnlockPickup-DecomposedReward-NoMission-NoReward-v0"
    "MiniGrid-PutNear-6x6-N2-v0"
    "MiniGrid-IsNextTo-6x6-v0"
    "MiniGrid-IsNextTo-6x6-DecomposedReward-v0"
    "MiniGrid-IsNextTo-6x6-DecomposedReward-NoMission-v0"
    "MiniGrid-IsNextTo-6x6-DecomposedReward-NoReward-v0"
    "MiniGrid-IsNextTo-6x6-DecomposedReward-NoMission-NoReward-v0"
    "MiniGrid-IsNextTo-6x6-OracleDecomposedReward-v0"
    "MiniGrid-IsNextTo-6x6-OracleDecomposedReward-NoMission-v0"
    "MiniGrid-IsNextTo-6x6-OracleDecomposedReward-NoReward-v0"
    "MiniGrid-IsNextTo-6x6-OracleDecomposedReward-NoMission-NoReward-v0"
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
echo "DEBUG_STR: $DEBUG_STR"

search_string() {
    local file="$1"
    local search_string="$2"

    if [[ -n $(cat $file | grep $search_string) ]]; then
        return 0
    else
        return 1
    fi
}


for env_id in ${ENV_IDS[@]}
do
    # Create default hparams for envs
    CONF_FILE="llm_curriculum/envs/minimal_minigrid/hyperparams/ppo_minigrid.yml"
    search_string "$CONF_FILE" "$env_id"
    result=$?
    echo "search_string result: $result"
    # Note: 0 is success, i.e. string was found
    # 1 is failure, i.e. string not found, so we should create that.
    if [[ $result -eq 1 ]]; then
        echo "Creating default hparams for $env_id"
        hparams="
$env_id: 
  <<: *minigrid-defaults
  n_timesteps: !!float 3e5"
        echo "$hparams" >> $CONF_FILE
    fi

    # Run experiment
    python llm_curriculum/envs/minimal_minigrid/train.py \
        --algo ppo $WANDB_STR $DEBUG_STR \
        --env $env_id \
        --conf-file $CONF_FILE \
        --record-video
done