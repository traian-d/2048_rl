from typing import Any

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from environment.Board import Board


class EnvStats:

    def __init__(self) -> None:
        self._reward_history = []
        self._step_count = 0
        self._max_reward = 0
        self._total_reward = 0

    def update_all(self, reward) -> None:
        self.increment_count()
        self.update_rewards(reward)

    def increment_count(self) -> None:
        self._step_count += 1

    def update_rewards(self, reward) -> None:
        self._reward_history += [reward]
        self._total_reward += reward
        if reward > self._max_reward:
            self._max_reward = reward

    @property
    def reward_history(self):
        return self._reward_history

    @property
    def step_count(self):
        return self._step_count

    @property
    def total_reward(self):
        return self._total_reward

    @property
    def max_reward(self):
        return self._max_reward


class Env2048(py_environment.PyEnvironment):

    def __init__(self, reward_func=None, discount=1.0, invalid_act_rew=-10):
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4, 4), dtype=np.int32, minimum=0, name='observation')
        self._board = Board(reward_func=reward_func, invalid_act_reward=invalid_act_rew)
        self._stats = EnvStats()
        self._discount = discount
        self._reward_func = reward_func
        self._invalid_act_rew = invalid_act_rew

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._board = Board(reward_func=self._reward_func, invalid_act_reward=self._invalid_act_rew)
        self._stats = EnvStats()
        return ts.restart(self._board.to_np())

    def _step(self, action):
        action_scalar = action.item(0)
        current_board, reward, done = self._board.step(action_scalar)
        if done:
            # TODO: see if this is a good idea
            # Rewards are not updated, because at this stage board will return total reward
            self._stats.increment_count()
            return ts.termination(current_board, reward)
        self._stats.update_all(reward)
        return ts.transition(current_board, reward=reward, discount=self._discount)

    def get_state(self) -> Any:
        """
        This implementation actually piggy-backs on the overridden method.
        It will return statistics, not the environment state.

        :return: Environment statistics
        """
        return self._stats


if __name__ == '__main__':
    from tf_agents.environments import tf_py_environment

    env = tf_py_environment.TFPyEnvironment(Env2048())
    # env = Env2048()

    rewards = []
    max_rewards = []
    steps = []
    total_rewards = []
    env.reset()

    for i in range(10):
        if not i % 100:
            print(i)
        time_step = env.step(np.array(np.random.randint(4), dtype=np.int32))
        while not time_step.is_last():
            time_step = env.step(np.array(np.random.randint(4), dtype=np.int32))
        if time_step.is_last():
            pyenv = env.pyenv.envs[0]
            #         pyenv = env
            rewards += pyenv.get_state().reward_history
            steps += [pyenv.get_state().step_count]
            total_rewards += [pyenv.get_state().total_reward]
            max_rewards += [pyenv.get_state().max_reward]
            env.reset()

