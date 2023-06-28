# LLM Curriculum

Code to run experiments on leveraging LLM's compositional task-solving capacity to accelerate robotic skill learning. 

## Quickstart

Set up environment
```bash
conda create -n llm-curriculum python=3.9
conda activate
```

Install dependencies:
```bash
python -m pip install setuptools==65.6.3 pip==21
python -m pip install -r requirements/base.txt
python -m pip install -r requirements/dev.txt
python -m pip install -e .
pre-commit install
```

Check tests work:
```bash
python -m pytest tests
```

## Examples

Original Fetch environment: `examples/fetch_pick_and_place.py`. 

Custom Fetch environment: `examples/fetch_custom_env.py`. 

## Training

Single-task training:
```bash
python llm_curriculum/learning/train_singletask.py
```

Multi-task learning:
```bash
python llm_curriculum/learning/train_multitask_separate.py
```

### Config system

Experiments can be run like this:

```bash
python llm_curriculum/learning/train_multitask_separate.py \
    --config llm_curriculum/learning/config/default.py
```

Hyperparameters can be overriden like this:

```bash
python llm_curriculum/learning/train_multitask_separate.py \
    --config llm_curriculum/learning/config/default.py \
    --config.do_track=False
```

Added a new hyperparamter help; when True, the program will exit after printing hparams. Useful for checking configs

```bash
# Print the hparams and exit
python llm_curriculum/learning/train_multitask_separate.py \
    --config llm_curriculum/learning/config/default.py \
    --config.help=True
```

## Documentation

Planning doc: https://docs.google.com/document/d/1e6-Mq5KjXKuIgxf6-E3fBbqIb5GeNbuYAngC6YDuEgQ/edit#heading=h.yyfnkspbactf
Slideshow: https://docs.google.com/presentation/d/1gTlCxZdajc9d1fsrd4AskrUWt8AACdulaWzVvB256_w/edit?usp=sharing
Prompting: https://github.com/dtch1997/gpt-text-gym.git
