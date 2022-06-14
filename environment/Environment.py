import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from environment.Board import Board


class Env2048(py_environment.PyEnvironment):

    def __init__(self, reward_func=None, discount=1.0, invalid_act_rew=-10):
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4, 4), dtype=np.int32, minimum=0, name='observation')
        self._board = Board(reward_func=reward_func, invalid_act_reward=invalid_act_rew)
        self._discount = discount

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._board = Board()
        return ts.restart(self._board.to_np())

    def _step(self, action):
        action_scalar = action.item(0)
        current_board, reward, done = self._board.step(action_scalar)
        if done:
            return ts.termination(current_board, reward)
        return ts.transition(current_board, reward=reward, discount=self._discount)


if __name__ == '__main__':
    environment = Env2048()
    utils.validate_py_environment(environment, episodes=5)

