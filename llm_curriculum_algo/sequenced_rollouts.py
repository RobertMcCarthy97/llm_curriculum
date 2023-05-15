from typing import Optional, Tuple, Dict, Union, List, Any, Callable
import numpy as np
import gym
import warnings

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv, unwrap_vec_normalize, is_vecenv_wrapped, DummyVecEnv, VecMonitor


class SequencedRolloutCollector():
    '''
    # TODO: Checks to perform:
    - correct model is sampling actions
    - correct data fed to model when sampling actions
    - normed obs working ion principled way
    - data stored in correct replay buffer
    - all models being updated, correct models being updated
    '''
    def __init__(self, env, models_dict):
        self.env = env
        self.models_dict = models_dict
        self._vec_normalize_env = unwrap_vec_normalize(env)
        assert len(self.env.envs) == 1

    def collect_rollouts(
        self,
        # env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        # replay_buffer: ReplayBuffer,
        # action_noise: Optional[ActionNoise] = None,
        # learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> Tuple[RolloutReturn, Dict[str, int]]:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        env = self.env # add to locals for callbacks...
        
        models_steps_taken = {}
        for model_name, model in self.models_dict.items():
            # init counter
            models_steps_taken[model_name] = 0
            
            # Switch to eval mode (this affects batch norm / dropout)
            model.policy.set_training_mode(False) # TODO: why is this False!!!!????

            # Vectorize action noise if needed
            assert model.action_noise is None, "not setup to deal with this yet"
            # if model.action_noise is not None and self.env.num_envs > 1 and not isinstance(model.action_noise, VectorizedActionNoise):
            #     action_noise = VectorizedActionNoise(action_noise, self.env.num_envs)
            #     assert False

            assert not model.use_sde, "not setup to deal with this yet"
            # if self.use_sde:
            #     model.actor.reset_noise(env.num_envs)
            
            assert self.env is model.env
            assert self._vec_normalize_env is model._vec_normalize_env
        
        assert isinstance(self.env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."
        if self.env.num_envs > 1:
                assert model.train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."
           
        num_collected_steps, num_collected_episodes = 0, 0
        
        self.rollout_reset()

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            
            # Select which model to use
            model, model_name = self.choose_model()
            
            # Since changing models, need to manually update models last obs (in case model has changed)
            self.update_last_obs_model(model)
            
            assert not model.use_sde, "Not implemented yet"
            # if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
            #     # Sample a new noise matrix
            #     self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = model._sample_action(model.learning_starts, model.action_noise, self.env.num_envs) # TODO: should treat learning starts differently here??

            assert self.env.envs[0].agent_conductor.active_task.name == model_name
            # Rescale and perform action
            new_obs, rewards, dones, infos = self.env.step(actions)
            assert model_name == infos[0]["prev_task_name"]
            '''
            Note: env resets automatically
            - new_obs is reset obs, but rewards, dones, info are from final step of previous episode
            - info contains the terminal obs for storing last transition
            - this is all dealt with appropriately in the model._store_transition method
            '''

            model.num_timesteps += self.env.num_envs
            models_steps_taken[model_name] += self.env.num_envs
            num_collected_steps += 1
            
            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * self.env.num_envs, num_collected_episodes, continue_training=False), models_steps_taken

            # Retrieve reward and episode length if using Monitor wrapper
            # TODO: fix logging
            model._update_info_buffer(infos, dones) # these are just used for SB3 logging - currently not working correct

            # Store data in replay buffer (normalized action and unnormalized observation)
            model._store_transition(model.replay_buffer, buffer_actions, new_obs, rewards, dones, infos)
            # Now manually record last obs - used by model later (model records last obs in store transition)
            self.update_last_obs(new_obs)

            model._update_current_progress_remaining(model.num_timesteps, model._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            model._on_step()

            # print(f"done: {dones[0]}")
            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    # model._episode_num += 1 # TODO: record episode when model swithces??

                    assert model.action_noise is None, "not setup to deal with this yet"
                    # if action_noise is not None:
                    #     kwargs = dict(indices=[idx]) if self.env.num_envs > 1 else {}
                    #     action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and model._episode_num % log_interval == 0:
                        # TODO: fix logging
                        model._dump_logs()
                    
        callback.on_rollout_end()
        
        return RolloutReturn(num_collected_steps * self.env.num_envs, num_collected_episodes, continue_training), models_steps_taken

    def rollout_reset(self):
        # TODO: triple check this manual reset isn't throwing anything off...
        self._last_obs = self.env.reset()  # pytype: disable=annotation-type-mismatch
        self._last_episode_starts = None # shouldn't need these in off-policy...
        # Retrieve unnormalized observation for saving into the buffer
        if self._vec_normalize_env is not None:
            self._last_original_obs = self._vec_normalize_env.get_original_obs()
        else:
            assert False, "must be vec_norm_env" # TODO: also need to check for resets within loop?
        # reset prev model
        self.prev_model = None
            
    def update_last_obs(self, new_obs):
        # copy what is done in model.store_transition (assuming vec_norm_env is not None)
        self._last_obs = new_obs
        if self._vec_normalize_env is not None:
            self._last_original_obs = self._vec_normalize_env.get_original_obs()
        else:
            assert False, "must be vec_norm_env"
            
    def update_last_obs_model(self, model):
        # copy what is done in model.store_transition (assuming vec_norm_env is not None)
        # make sure to do this between model selection and model action sampling
        model._last_obs = self._last_obs
        model._last_original_obs = self._last_original_obs
            
    def choose_model(self):
        active_task = self.env.envs[0].agent_conductor.get_active_task() # TODO: just delegate model selection to agent conductor like this?
        model_name = active_task.name
        model = self.models_dict[model_name]
        # record model change
        if model is not self.prev_model:
            model._episode_num += 1
            self.prev_model = model
        return model, model_name

   
def evaluate_sequenced_policy(
    models_dict,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    def choose_model(env, models_dict):
        '''
        # TODO: shouldn't be duplicating code like this!
        '''
        active_task = env.envs[0].agent_conductor.get_active_task() # TODO: just delegate model selection to agent conductor like this?
        model_name = active_task.name
        model = models_dict[model_name]
        return model, model_name
    
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        # Select which model to use
        model, model_name = choose_model(env, models_dict)
        # rollout
        actions, states = model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
