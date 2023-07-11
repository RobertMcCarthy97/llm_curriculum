from huggingface_sb3 import EnvironmentName
from rl_zoo3.exp_manager import ExperimentManager
from stable_baselines3.common.vec_env import VecEnv
from llm_curriculum.envs.minimal_minigrid.sb3.callbacks import VideoRecorderCallback


class MyExperimentManager(ExperimentManager):
    def __init__(self, *args, **kwargs):
        eval_env_id = kwargs.pop("eval_env_id", None)
        record_video = kwargs.pop("record_video", False)
        super().__init__(*args, **kwargs)
        if eval_env_id is not None:
            self.eval_env_name = EnvironmentName(eval_env_id)
        else:
            self.eval_env_name = self.env_name
        self.record_video = record_video

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

    def create_callbacks(self):
        super().create_callbacks()
        if not self.record_video:
            return

        # Overwrite env kwargs to create env with render_mode="rgb_array"
        self.orig_env_kwargs = self.env_kwargs.copy()
        self.env_kwargs["render_mode"] = "rgb_array"
        video_callback = VideoRecorderCallback(
            self.create_envs(1, eval_env=True),
            render_freq=10000 // self.n_envs,
        )
        self.env_kwargs = self.orig_env_kwargs
        self.callbacks.append(video_callback)
