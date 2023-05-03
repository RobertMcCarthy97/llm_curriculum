# llm_envs/setup.py
from setuptools import setup, find_packages

setup(
    name="llm_curriculum_algo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'mujoco==2.3.3',
        'gymnasium==0.28.1',
    ],
)
