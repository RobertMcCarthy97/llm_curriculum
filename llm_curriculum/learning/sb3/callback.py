import gym
import torch as th
import numpy as np
from PIL import Image, ImageDraw

from typing import Any, Dict
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        render_freq: int,
        n_eval_episodes: int = 1,
        add_text: bool = False,
        sequenced_rollouts: bool = False,
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self.add_text = add_text
        self.sequenced_rollouts = sequenced_rollouts

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            self._record_video(deterministic=True)
            self._record_video(deterministic=False)
        return True

    def _record_video(self, deterministic=True):
        screens = []
        behaviour_str = "deterministic" if deterministic else "stochastic"

        def overlay_text(rgb_array: np.array, info: Dict[str, Any] = None):
            # import pdb; pdb.set_trace()
            # Convert the RGB array to a PIL image
            img = Image.fromarray(rgb_array.astype(np.uint8))
            # Call draw Method to add 2D graphics in an image
            I1 = ImageDraw.Draw(img)
            # Add Text to an image
            if "active_task_name" in info:
                I1.text(
                    (28, 36),
                    f"Active task: {info['active_task_name']}",
                    fill=(255, 0, 0),
                )
            I1.text((28, 50), f"Success: {info['is_success']}", fill=(255, 0, 0))
            # Convert the PIL image back to an RGB array
            rgb_array_with_text = np.array(img).astype(np.uint8)
            return rgb_array_with_text

        def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
            """
            Renders the environment in its current state, recording the screen in the captured `screens` list

            :param _locals: A dictionary containing all local variables of the callback's scope
            :param _globals: A dictionary containing all global variables of the callback's scope
            """
            screen = self._eval_env.render(mode="rgb_array")
            if self.add_text:
                assert len(self._eval_env.envs) == 1, "not setup for more than 1 env!"
                screen = overlay_text(screen, _locals["infos"][0])
            # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
            screens.append(screen.transpose(2, 0, 1))

        if self.sequenced_rollouts:
            raise NotImplementedError()
        else:
            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=deterministic,
            )
        self.logger.record(
            f"trajectory/video_{behaviour_str}",
            Video(th.ByteTensor(np.array([screens])), fps=25),
            exclude=("stdout", "log", "json", "csv"),
        )
