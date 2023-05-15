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
from sb3_callbacks import SuccessCallback, VideoRecorderCallback, EvalCallbackCustom


def create_env(hparams):
    # Create env
    if hparams['use_baseline_env']:
        env = make_env_baseline("FetchReach-v2", render_mode=hparams['render_mode'], max_ep_len=hparams['max_ep_len'])
        hparams['replay_buffer_class'] = None
        hparams['info_keywords'] = ("is_success",)
    else:
        env = make_env(
            manual_decompose_p=hparams['manual_decompose_p'],
            dense_rew_lowest=hparams['dense_rew_lowest'],
            use_language_goals=hparams['use_language_goals'],
            render_mode=hparams['render_mode'],
            max_ep_len=hparams['max_ep_len'],
            single_task_names=hparams['single_task_names'],
            high_level_task_names=hparams['high_level_task_names'],
            contained_sequence=hparams['contained_sequence'],
            )

    # Vec Env
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, info_keywords=hparams['info_keywords'])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    return env

'''
conda activate llm_curriculum
''' 

if __name__ == "__main__":
    '''
    TODO: implement W&B sweep?
    '''
    
    hparams = {
        'seed': 0,
        # env
        'manual_decompose_p': 1,
        'dense_rew_lowest': True,
        'use_language_goals': False,
        'render_mode': 'rgb_array',
        'use_oracle_at_warmup': False,
        'max_ep_len': 50,
        'use_baseline_env': False,
        # task
        'single_task_names': ['move_gripper_to_cube'],
        'high_level_task_names': ['move_cube_to_target'],
        'contained_sequence': False,
        # algo
        'algo': TD3, # DDPG/TD3/SAC
        'policy_type': 'MultiInputPolicy',
        'learning_starts': 1e3,
        'replay_buffer_class': None, # LLMBasicReplayBuffer , None
        'replay_buffer_kwargs': None, # None, {'keep_goals_same': True, 'do_parent_relabel': True, 'parent_relabel_p': 0.2}
        'total_timesteps': 1e5,
        'device': 'cpu',
        # logging
        'do_track': True,
        'log_path': "./logs/" + f"{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}",
        'exp_name': 'move_gripper_to_cube-dense_reward',
        'exp_group': 'dense_reward',
        'info_keywords': ('is_success', 'overall_task_success', 'active_task_level'),
    }
    
    # W&B
    if hparams['do_track']:
        run = wandb.init(
            entity='robertmccarthy11',
            project='llm-curriculum',
            group=hparams['exp_group'],
            name=hparams['exp_name'],
            job_type='training',
            # config=vargs,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=False,  # auto-upload the videos of agents playing the game
            save_code=False,  # optional
        )
    
    # Seed
    set_random_seed(hparams['seed'])  # type:ignore

    # create envs
    env = create_env(hparams)
    eval_env = create_env(hparams)

    # Logger and callbacks
    logger = configure(hparams['log_path'], ["stdout", "csv", "tensorboard"])
    callback_list = []
    callback_list += [EvalCallbackCustom(eval_env, eval_freq=1000, best_model_save_path=None)] # TODO: use different eval env
    if not hparams['use_baseline_env']:
        callback_list += [SuccessCallback(log_freq=1000)]
    callback_list += [VideoRecorderCallback(env, render_freq=10000, n_eval_episodes=1, add_text=True)]
    # wandb
    if hparams['do_track']:
        # log hyperparameters
        wandb.log({'hyperparameters': hparams})
        # wandb callback
        callback_list += [
            WandbCallback(
                gradient_save_freq=10,
                model_save_path=None,
                verbose=1,
            )]
    callback = CallbackList(callback_list)

    # Algo
    model = hparams['algo'](
                hparams['policy_type'],
                env,
                verbose=1,
                learning_starts=hparams['learning_starts'],
                replay_buffer_class=hparams['replay_buffer_class'],
                replay_buffer_kwargs=hparams['replay_buffer_kwargs'],
                device=hparams['device'],
                use_oracle_at_warmup=hparams['use_oracle_at_warmup'],
                )
    model.set_logger(logger)

    # Train the model
    model.learn(total_timesteps=hparams['total_timesteps'], callback=callback, progress_bar=False)
    
    if hparams['do_track']:
        run.finish()
