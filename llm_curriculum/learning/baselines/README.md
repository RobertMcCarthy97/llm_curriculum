## Instructions

Run vanilla TD3: 
```bash
python llm_curriculum/learning/baselines/vanilla_rl.py \
    --config llm_curriculum/baselines/config/default.py
```

Run TD3 + HER:
```bash
python llm_curriculum/learning/baselines/vanilla_rl.py \
    --config llm_curriculum/baselines/config/default.py \
    --config.use_her=True
```

## Errors

Custom replay buffer enforces some conditions on multi-task learning

```bash 
$ python llm_curriculum/learning/baselines/vanilla_rl.py --config llm_curriculum/learning/config/default.py --config.wandb.track=False
...
File "/home/daniel/anaconda3/envs/llm-curriculum-test/lib/python3.9/site-packages/stable_baselines3/common/buffers_custom.py", line 382, in add
    assert all([info['prev_task_name'] == self.task_name for info in infos]), "Task name mismatch"
AssertionError: Task name mismatch
```

`use_baseline_env` can't be set to True because it results in an error

```bash
$ python llm_curriculum/learning/train_multitask_separate.py --config llm_curriculum/learning/config/default.py --config.wandb.track=False --config.use_baseline_env=True
...
AttributeError: 'MujocoFetchReachEnv' object has no attribute 'agent_conductor'
```