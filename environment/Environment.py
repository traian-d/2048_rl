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
        self._observation_spec = array_spec.ArraySpec(
            shape=(16, ), dtype=np.int32, name='observation')
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
    from tf_agents.networks import sequential
    from tf_agents.agents.dqn import dqn_agent
    from tf_agents.specs import tensor_spec
    from tf_agents.utils import common

    import tensorflow as tf
    import numpy as np

    num_iterations = 20000 # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration = 1# @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}

    train_env = tf_py_environment.TFPyEnvironment(Env2048())

    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    # q_net.build(input_shape=(5,5))
    # print(q_net.summary())

    agent = dqn_agent.DqnAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    # from tf_agents.environments import tf_py_environment
    #
    # env = tf_py_environment.TFPyEnvironment(Env2048())
    # # env = Env2048()
    #
    # rewards = []
    # max_rewards = []
    # steps = []
    # total_rewards = []
    # env.reset()
    #
    # for i in range(10):
    #     if not i % 100:
    #         print(i)
    #     time_step = env.step(np.array(np.random.randint(4), dtype=np.int32))
    #     while not time_step.is_last():
    #         time_step = env.step(np.array(np.random.randint(4), dtype=np.int32))
    #     if time_step.is_last():
    #         pyenv = env.pyenv.envs[0]
    #         #         pyenv = env
    #         rewards += pyenv.get_state().reward_history
    #         steps += [pyenv.get_state().step_count]
    #         total_rewards += [pyenv.get_state().total_reward]
    #         max_rewards += [pyenv.get_state().max_reward]
    #         env.reset()

