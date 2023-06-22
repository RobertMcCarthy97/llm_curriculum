import numpy as np
import gymnasium as gym
from llm_curriculum.envs.agent_conductor import AgentConductor


class CurriculumEnvWrapper(gym.Wrapper):
    """
    - Adds the target pos to observation
    - Goals are specified in language, and decomposed by agent_conductor
    """

    def __init__(
        self,
        env,
        agent_conductor,
        use_language_goals=False,
        max_ep_len=50,
        drawer_env=False,
    ):
        super().__init__(env)
        self._env = env
        self.use_language_goals = use_language_goals
        self.agent_conductor = agent_conductor
        self.max_ep_len = max_ep_len
        self.drawer_env = drawer_env

        self.init_obs_space()
        self.reset_conductor = AgentConductor(
            env,
            manual_decompose_p=1,
            high_level_task_names=agent_conductor.high_level_task_names,
        )  # TODO: what if using different high-level task
        self.set_state_parsers()

        # begin counter
        self.active_task_steps = 0
        self.episode_n_steps = 0

    def init_obs_space(self):
        # goal observation
        if self.use_language_goals:
            goal_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(384,), dtype=np.float32
            )
        else:
            # assert False, "Not working for direction tasks"
            n_tasks = self.agent_conductor.n_tasks
            goal_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(n_tasks,), dtype=np.float32
            )
        # new dict observation space
        self.observation_space = gym.spaces.Dict(
            {
                "observation": self._env.observation_space,
                "desired_goal": goal_space,
            }
        )

    def set_state_parsers(self):
        # TODO: this should just be done when state_parsers are initted
        for conductor in [self.agent_conductor, self.reset_conductor]:
            for task in conductor.high_level_task_list:
                # init parsers
                task.set_state_parser_full_tree("drawer" if self.drawer_env else "core")
                # check shapes
                task.state_parser.check_obs_space(self.observation_space["observation"])

    def reset(self, **kwargs):
        prev_task = self.agent_conductor.get_active_task()
        # single task scenario
        if len(self.agent_conductor.single_task_names) > 0:
            obs, info = self.reset_single_task(
                **kwargs
            )  # TODO: need to stop task info tracking??
        else:
            obs, info = self.reset_normal(**kwargs)
        # stats
        self.episode_n_steps = 0
        active_task = self.agent_conductor.get_active_task()
        self.record_task_stats(prev_task, active_task, reset=True)
        assert active_task.name == info["active_task_name"]
        # return
        return obs, info

    def reset_normal(self, **kwargs):
        # reset env
        state_obs, _ = self._env.reset(**kwargs)
        self.active_state_obs = state_obs
        # get task
        active_task = self.agent_conductor.reset()
        # obs
        obs = self.set_obs(state_obs, active_task)
        # info
        info = self.set_info(active_task, active_task, state_obs)

        return obs, info

    def reset_single_task(self, **kwargs):
        """
        Use agent oracle actions to step through subtask until reached the single task

        TODO: extend as follows:
        - take 'task_to_reset_to' as input
        - replace self.agent_conductor.active_single_task_name with 'task_to_reset_to'
        - 'task_to_reset_to' can just be high_level_task_name if no single tasks???
        """
        for _ in range(50):
            # reset env
            obs, info = self.reset_normal(**kwargs)
            # decide task to reset to
            single_task_name = self.agent_conductor.get_active_single_task_name()
            single_task = self.agent_conductor.get_task_from_name(single_task_name)
            reset_task = self.agent_conductor.decide_initial_state_curriculum_task(
                single_task
            )
            # reset reset_conductor
            self.reset_conductor.set_single_task_names([reset_task.name])
            self.reset_conductor.reset()
            for _ in range(50):
                # check if reached desired task
                if self.reset_conductor.get_active_task().name == reset_task.name:
                    # if so, return
                    return obs, info
                reset_prev_active_task = self.reset_conductor.get_active_task()
                # action and step env
                action = self.reset_conductor.get_oracle_action(
                    obs["observation"], reset_prev_active_task
                )
                obs, _, _, truncated, info = self.step(
                    action, reset_step=True
                )  # Don't record till reset
                _, reset_success = self.calc_reward(
                    obs["observation"], reset_prev_active_task
                )
                # step reset_conductor
                reset_active_task = self.reset_conductor.step()
                reset_goal_changed = (
                    reset_prev_active_task.name != reset_active_task.name
                )
                if (reset_success and reset_goal_changed) or truncated:
                    self.reset_conductor.record_task_success_stat(
                        reset_prev_active_task, reset_success
                    )
        assert False, "Failed to reach single task"

    def step(self, action, reset_step=False):
        # TODO: fix reward? currently r = f(s_t+1, g_t)
        # prev active task
        prev_active_task = self.agent_conductor.get_active_task()

        # step env
        state_obs, _, _, _, _ = self._env.step(action)
        self.active_state_obs = state_obs

        # calc reward
        reward, success = self.calc_reward(
            state_obs, prev_active_task
        )  # reward based on g_t, obs_t+1

        # replan
        active_task = self.agent_conductor.step()

        # obs
        obs = self.set_obs(state_obs, active_task)

        # info
        info = self.set_info(prev_active_task, active_task, state_obs)
        info["is_success"], info["success"] = success, success
        info["goal_changed"] = prev_active_task.name != active_task.name
        assert active_task.name == info["active_task_name"]

        if not reset_step:
            self.episode_n_steps += 1
        # terminated
        terminated = False
        # truncated
        truncated = self.episode_n_steps >= self.max_ep_len

        # stats
        if not reset_step:
            # record stats
            self.record_task_stats(prev_active_task, active_task)
            if (success and info["goal_changed"]) or truncated:
                self.agent_conductor.record_task_success_stat(prev_active_task, success)

        return obs, reward, terminated, truncated, info

    def record_task_stats(self, prev_task, active_task, reset=False):
        # check if task changed
        task_changed = prev_task.name != active_task.name
        if (task_changed or reset) and (self.active_task_steps > 0):
            # record previous task count and n_steps
            self.agent_conductor.record_task_chosen_stat(
                prev_task, self.active_task_steps
            )
            # reset steps
            self.active_task_steps = 0
        else:
            # increment steps
            self.active_task_steps += 1

    def set_obs(self, state_obs, active_task):
        obs = {}
        obs["observation"] = state_obs
        # update desired goal in obs
        obs["desired_goal"] = self.get_task_goal(active_task)
        # TODO: implement achieved goal...
        return obs

    def get_task_goal(self, task):
        if self.use_language_goals:
            return self.agent_conductor.get_task_embedding(task)
        else:
            return self.agent_conductor.get_task_oracle_goal(task)

    def set_info(self, prev_task, stepped_task, current_state):
        info = {}
        # prev task details
        info["prev_task_level"] = prev_task.level
        info["prev_task_name"] = prev_task.name
        # active task details
        info["active_task_level"] = stepped_task.level
        info["active_task_name"] = stepped_task.name
        # info['active_task_index'] = self.agent_conductor.get_task_index(stepped_task)
        # record parent tasks details
        iter_task = stepped_task
        for i in range(stepped_task.level, -1, -1):
            info[f"task_level_{i}"] = iter_task.name
            iter_task = iter_task.parent_task
        # record parent task reward and goal
        if prev_task.parent_task is not None:  # TODO: delete this?
            info["obs_parent_goal_reward"] = prev_task.parent_task.check_success_reward(
                current_state
            )[
                1
            ]  # parent reward depends on parent_g_t, obs_t+1 - TODO: this naming is very confusding!!
            info["obs_parent_goal"] = self.get_task_goal(prev_task.parent_task)
        for parent_name in prev_task.relations["parents"].keys():
            info[f"obs_{parent_name}_reward"] = self.agent_conductor.get_task_from_name(
                parent_name
            ).check_success_reward(current_state)[1]
        # record overall success
        (
            info["overall_task_success"],
            _,
        ) = self.agent_conductor.chosen_high_level_task.check_success_reward(
            current_state
        )
        # return
        return info

    def calc_reward(self, state, active_task):
        success, reward = active_task.active_task_check_and_set_success(state)
        return reward, success

    def plot_state(self):
        # plot the state of the environment, for debugging purposes
        pass

    def get_oracle_action(self, state):
        return self.agent_conductor.get_oracle_action(
            state, self.agent_conductor.active_task
        )
