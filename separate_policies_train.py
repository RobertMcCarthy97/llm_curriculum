from datetime import datetime

from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CallbackList, EveryNTimesteps
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.type_aliases import MaybeCallback

import wandb
from wandb.integration.sb3 import WandbCallback

from env_wrappers import make_env, make_env_baseline
from stable_baselines3.common.buffers_custom import LLMBasicReplayBuffer
from sb3_callbacks import SuccessCallback, VideoRecorderCallback, EvalCallbackCustom, SuccessCallbackMultiRun


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
            # TODO: remove goal info from obs
            )
        # assert False, "remove goal info from obs" # TODO

    # Vec Env
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, info_keywords=hparams['info_keywords'])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    return env

def setup_logging(env, hparams):
    # Logger and callbacks
    logger = configure(hparams['log_path'], ["stdout", "csv", "tensorboard"])
    callback_list = []
    # callback_list += [EvalCallbackCustom(eval_env, eval_freq=1000, best_model_save_path=None)] # TODO: use different eval env
    if not hparams['use_baseline_env']:
        callback_list += [SuccessCallbackMultiRun(log_freq=1000*len(hparams['single_task_names']))] # TODO: dodgy freq setting
    # callback_list += [VideoRecorderCallback(env, render_freq=10000, n_eval_episodes=1, add_text=True)]
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
    return logger, callback
    
def create_models(env):
    task_list = env.envs[0].agent_conductor.get_single_task_names()
    assert len(task_list) > 0
    
    models_dict = {}
    
    # create models
    for task in task_list:
        model = hparams['algo'](
                    hparams['policy_type'],
                    env,
                    verbose=1,
                    learning_starts=hparams['learning_starts'],
                    replay_buffer_class=hparams['replay_buffer_class'], # TODO: add proper replay buffer
                    replay_buffer_kwargs=hparams['replay_buffer_kwargs'],
                    device=hparams['device'],
                    use_oracle_at_warmup=hparams['use_oracle_at_warmup'],
                    )
        model.set_logger(logger)
        models_dict[task] = model
        
    # # link child to parent buffers
    # for task in models_dict.keys():
    #     if len(task.subtask_sequence) > 0:
    #         for subtask in task.subtask_sequence:
    #             subtask_name = subtask.name
    #             subtask_buffer = models_dict[subtask_name].replay_buffer
    #             models_dict[task].replay_buffer.add_child_buffer(subtask_buffer)
                
    return models_dict

def init_training(
        models_dict,
        total_timesteps: int,
        callback: MaybeCallback = None,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
    
    for _, model in models_dict.items():
        total_timesteps, callback = model._setup_learn(
                total_timesteps,
                callback,
                reset_num_timesteps,
                tb_log_name,
                progress_bar,
            )
        callback.on_training_start(locals(), globals())
      
def training_loop(models_dict, env, total_timesteps, log_interval=4):
    tasks = list(models_dict.keys())
    while models_dict[tasks[0]].num_timesteps < total_timesteps:
    
        for task_name, model in models_dict.items():
            # Set task
            for vec_env in env.envs:
                vec_env.agent_conductor.set_single_task_names([task_name])
            print("set task to", task_name)
            
            # Collect data
            rollout = model.collect_rollouts(
                    env,
                    train_freq=model.train_freq,
                    action_noise=model.action_noise,
                    callback=callback,
                    learning_starts=model.learning_starts,
                    replay_buffer=model.replay_buffer,
                    log_interval=log_interval,
                )

            if model.num_timesteps > 0 and model.num_timesteps > model.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = model.gradient_steps if model.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    model.train(batch_size=model.batch_size, gradient_steps=gradient_steps)


def get_hparams():
    return {
        'seed': 0,
        # env
        'manual_decompose_p': 1,
        'dense_rew_lowest': False,
        'use_language_goals': False,
        'render_mode': 'rgb_array',
        'use_oracle_at_warmup': False,
        'max_ep_len': 50,
        'use_baseline_env': False,
        # task
        'single_task_names': ['lift_cube', 'cube_between_grippers', 'grasp_cube', 'move_gripper_to_cube'],
        'high_level_task_names': ['move_cube_to_target'],
        'contained_sequence': False,
        # algo
        'algo': TD3, # DDPG/TD3/SAC
        'policy_type': 'MultiInputPolicy', # TODO: switch to non-dict
        'learning_starts': 1e3,
        'replay_buffer_class': None, # LLMBasicReplayBuffer , None
        'replay_buffer_kwargs': None, # None, {'keep_goals_same': True, 'do_parent_relabel': True, 'parent_relabel_p': 0.2}
        'total_timesteps': 1e5,
        'device': 'cpu',
        # logging
        'do_track': True,
        'log_path': "./logs/" + f"{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}",
        'exp_name': 'temp',
        'exp_group': 'temp',
        'info_keywords': ('is_success', 'overall_task_success', 'active_task_level'),
    }

if __name__ == "__main__":
    '''
    TODO: implement W&B sweep?
    '''
    
    hparams = get_hparams()
    
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

    # setup logging
    logger, callback = setup_logging(env, hparams)

    # create models
    models_dict = create_models(env)

    # Train
    init_training(models_dict, hparams['total_timesteps'], callback=callback)
    training_loop(models_dict, env, hparams['total_timesteps'])
    
    if hparams['do_track']:
        run.finish()


# create env

# create dict of models for each task

# link child buffers to parent buffers

# Train loop