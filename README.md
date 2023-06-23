# LLM Curriculum

Code to run experiments on leveraging LLM's compositional task-solving capacity to accelerate robotic skill learning. 

## Quickstart

Set up environment
```
make install
```

Run tests
```
make test
```

## Examples

Original Fetch environment: `examples/fetch_pick_and_place.py`. 

Custom Fetch environment: `examples/fetch_custom_env.py`. 

## Training

Single-task training:
```bash
python llm_curriculum/learning/sb3/train_singletask.py
```

Multi-task learning:
```bash
python llm_curriculum/learning/sb3/train_multitask_separate.py
```