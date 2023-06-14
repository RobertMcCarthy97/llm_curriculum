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
```

Check tests work:
```bash
python -m pytest tests
```

## Examples

Original Fetch environment: `examples/fetch_pick_and_place.py`. 
Custom Fetch environment: `examples/fetch_custom_env.py`. 

