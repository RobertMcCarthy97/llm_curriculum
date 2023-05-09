from typing import Optional

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv, unwrap_vec_normalize


class SequencedRolloutCollector():
    def __init__(self, env, models_dict):
        self.env = env
        self.models_dict = models_dict
        
        self._vec_normalize_env = unwrap_vec_normalize(env)
        assert False, "check correct..."
        assert len(self.env.envs) == 1

    def collect_rollouts(
        self,
        # env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
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
        # Switch to eval mode (this affects batch norm / dropout)
        # self.policy.set_training_mode(False) # TODO
        for model in self.models_dict.values():
            model.policy.set_training_mode(False) # TODO: why is this False!!!!????

        num_collected_steps, num_collected_episodes = 0, 0 # TODO

        assert isinstance(self.env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        import pdb; pdb.set_trace() # check if action_noise...
        if action_noise is not None and self.env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, self.env.num_envs)

        # if self.use_sde:
        #     self.actor.reset_noise(env.num_envs)
            
        self.rollout_reset()

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            
            # Select which model to use
            model = self.choose_model()
            # Since changing models, need to manually update models last obs (in case model has changed)
            self.update_last_obs_model(model)
            
            assert not model.use_sde, "Not implemented yet"
            # if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
            #     # Sample a new noise matrix
            #     self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = model._sample_action(learning_starts, action_noise, self.env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = self.env.step(actions)

            model.num_timesteps += self.env.num_envs
            num_collected_steps += 1
            
            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * self.env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            model._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            model._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)
            # Now manually record last obs - used by model later (model records last obs in store transition)
            self.update_last_obs(new_obs)

            model._update_current_progress_remaining(model.num_timesteps, model._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            model._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    model._episode_num += 1 # TODO: record episode when model swithces??
                    assert False, "Not implemented yet - need for logging" # maybe just do via this class??

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and model._episode_num % log_interval == 0:
                        model._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * self.env.num_envs, num_collected_episodes, continue_training)

    def rollout_reset(self):
        # TODO: triple check this manual reset isn't throwing anything off...
        self._last_obs = self.env.reset()  # pytype: disable=annotation-type-mismatch
        self._last_episode_starts = None # shouldn't need these in off-policy...
        # Retrieve unnormalized observation for saving into the buffer
        if self._vec_normalize_env is not None:
            self._last_original_obs = self._vec_normalize_env.get_original_obs()
        else:
            assert False, "must be vec_norm_env" # TODO: also need to check for resets within loop?
            
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
        model = self.models_dict[active_task]
        assert self.env is model.env
        return model