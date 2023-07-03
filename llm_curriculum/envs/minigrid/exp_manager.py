from huggingface_sb3 import EnvironmentName
from rl_zoo3.exp_manager import ExperimentManager
from stable_baselines3.common.vec_env import VecEnv


class MyExperimentManager(ExperimentManager):
    def __init__(self, *args, **kwargs):
        eval_env_id = kwargs.pop("eval_env_id", None)
        super().__init__(*args, **kwargs)
        if eval_env_id is not None:
            self.eval_env_name = EnvironmentName(eval_env_id)
        else:
            self.eval_env_name = self.env_name

    def create_envs(
        self, n_envs: int, eval_env: bool = False, no_log: bool = False
    ) -> VecEnv:
        """Create environments"""
        if eval_env:
            self.true_env_name = self.env_name
            self.env_name = self.eval_env_name
        env = super().create_envs(n_envs, eval_env, no_log)
        if eval_env:
            self.env_name = self.true_env_name
        return env
