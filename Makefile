.venv:
	python -m venv .venv
	# Pin setuptools and pip to avoid issues with installing gym
	.venv/bin/python -m pip install setuptools==65.6.3 pip==21
	# Pin wheel to 0.38.4 to avoid this: 
	# https://github.com/openai/gym/issues/3202 
	.venv/bin/python -m pip install wheel==0.38.4
	.venv/bin/python -m pip install pip-tools
	touch .venv

requirements/base.txt: requirements/base.in .venv
	.venv/bin/python -m piptools compile requirements/base.in -o requirements/base.txt
	touch requirements/dev.txt

requirements/dev.txt: requirements/dev.in .venv
	.venv/bin/python -m piptools compile requirements/dev.in -o requirements/dev.txt
	touch requirements/dev.txt

compile: requirements/base.txt requirements/dev.txt

.install_requires:
	.venv/bin/python -m pip install -r requirements/base.txt
	.venv/bin/python -m pip install -r requirements/dev.txt
	.venv/bin/python -m pip install -e .
	pre-commit install

test: 
	.venv/bin/python -m pytest -m pytest tests --cov=llm_curriculum --cov-report=xml

install: .venv compile .install_requires

all: install test

.PHONY: .install_requires compile install test all