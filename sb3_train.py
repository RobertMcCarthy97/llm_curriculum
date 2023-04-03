from datetime import datetime

from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CallbackList, EveryNTimesteps
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback

import wandb
from wandb.integration.sb3 import WandbCallback

from env_wrappers import make_env, make_env_baseline
from stable_baselines3.common.buffers_custom import LLMBasicReplayBuffer
from sb3_callbacks import SuccessCallback, VideoRecorderCallback

if __name__ == "__main__":
    '''
    TODO: implement W&B sweep?
    '''
    
    ##### hyperparams
    seed = 0
    # env
    manual_decompose_p = 1
    dense_rew_lowest = True
    use_language_goals = False
    render_mode = "rgb_array"
    oracle_n_action_seed = 0
    max_ep_len = 50
    use_baseline_env = False
    single_task_name = None
    # algo
    algo = DDPG # or TD3, SAC
    policy_type = "MultiInputPolicy"
    learning_starts = 1e2
    replay_buffer_class = LLMBasicReplayBuffer
    total_timesteps = 1e5
    device = 'cpu'
    # logging
    do_track = False
    log_path = "./logs/" + f"{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}"
    exp_name = "temp"
    exp_group = "temp"
    info_keywords = ("is_success", "overall_task_success", "active_task_level",)
    
    # seed
    set_random_seed(seed)  # type:ignore

    # create env
    
    if use_baseline_env:
        env = make_env_baseline("FetchReach-v2", render_mode=render_mode, max_ep_len=max_ep_len, single_task_name=single_task_name)
        replay_buffer_class = None
        info_keywords = ("is_success",)
    else:
        env = make_env(
        manual_decompose_p=manual_decompose_p,
        dense_rew_lowest=dense_rew_lowest,
        use_language_goals=use_language_goals,
        render_mode=render_mode,
        oracle_n_action_seed=oracle_n_action_seed,
        max_ep_len=max_ep_len,
        )
    
    # Vec Env
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, info_keywords=info_keywords)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Logger and callbacks
    logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    callback_list = []
    callback_list += [EvalCallback(env, eval_freq=1000, best_model_save_path=None)] # TODO: use different eval env
    if not use_baseline_env:
        callback_list += [SuccessCallback(log_freq=100)]
    callback_list += [VideoRecorderCallback(env, render_freq=10000, n_eval_episodes=1, add_text=True)]
    if do_track:
        callback_list += [
            WandbCallback(
                gradient_save_freq=10,
                model_save_path=None,
                verbose=1,
            )]
        run = wandb.init(
            entity='robertmccarthy11',
            project='llm-curriculum',
            group=exp_group,
            name=exp_name,
            job_type='training',
            # config=vargs,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=False,  # auto-upload the videos of agents playing the game
            save_code=False,  # optional
        )
    callback = CallbackList(callback_list)
    
    # Algo
    model = algo(policy_type, env, verbose=1, learning_starts=learning_starts, replay_buffer_class=replay_buffer_class, device=device)
    model.set_logger(logger)
    
    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    