
import numpy as np

from typing import Any
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import q_policy, greedy_policy, epsilon_greedy_policy, boltzmann_policy


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


class LocalDQN(dqn_agent.DqnAgent):

    def _setup_policy(self, time_step_spec, action_spec,
                      boltzmann_temperature, emit_log_probability):

        policy = q_policy.QPolicy(
            time_step_spec,
            action_spec,
            q_network=self._q_network,
            emit_log_probability=emit_log_probability,
            observation_and_action_constraint_splitter=(
                self._observation_and_action_constraint_splitter))

        if boltzmann_temperature is not None:
            collect_policy = boltzmann_policy.BoltzmannPolicy(
                policy, temperature=self._boltzmann_temperature)
        else:
            collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
                policy, epsilon=self._epsilon_greedy)
        policy = epsilon_greedy_policy.EpsilonGreedyPolicy(policy, epsilon=0.1)

        # Create self._target_greedy_policy in order to compute target Q-values.
        target_policy = q_policy.QPolicy(
            time_step_spec,
            action_spec,
            q_network=self._target_q_network,
            observation_and_action_constraint_splitter=(
                self._observation_and_action_constraint_splitter))
        self._target_greedy_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(target_policy, epsilon=0.1)

        return policy, collect_policy


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

    import tensorflow as tf

    from tf_agents.agents.dqn import dqn_agent
    from tf_agents.environments import tf_py_environment
    from tf_agents.networks import q_network
    from tf_agents.policies import random_tf_policy
    from tf_agents.replay_buffers import tf_uniform_replay_buffer
    from tf_agents.trajectories import trajectory
    from tf_agents.utils import common

    from matplotlib import pyplot as plt

    ## Hyperparameters

    num_iterations = 12000 # @param {type:"integer"}

    initial_collect_steps = 10000  # @param {type:"integer"}
    collect_steps_per_iteration = 10  # @param {type:"integer"}
    replay_buffer_capacity = 100000  # @param {type:"integer"}

    fc_layer_params = (100, 50)

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    gamma = 0.99
    log_interval = 200  # @param {type:"integer"}

    # num_atoms = 51  # @param {type:"integer"}
    # min_q_value = -10  # @param {type:"integer"}
    # max_q_value = 10  # @param {type:"integer"}
    n_step_update = 10  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 100  # @param {type:"integer"}

    ## Environment

    train_py_env = Env2048(reward_func=lambda x: x, invalid_act_rew=-10)
    eval_py_env = Env2048(reward_func=lambda x: x, invalid_act_rew=-10)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        #     num_atoms=num_atoms,
        fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        #     min_q_value=min_q_value,
        #     max_q_value=max_q_value,
        n_step_update=n_step_update,
        td_errors_loss_fn=common.element_wise_huber_loss,
        gamma=gamma,
        epsilon_greedy=0.5,
        train_step_counter=train_step_counter)
    agent.initialize()

    ## Metrics and Evaluation

    def compute_avg_return(environment, policy, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0
            rewards = []
            steps = []
            # while not time_step.is_last():
            for _ in range(200):
                action_step = policy.action(time_step)
                steps += [action_step.action.numpy()[0]]
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
                rewards += [time_step.reward.numpy()[0]]
            total_return += episode_return
            #     print(rewards)
            # print(steps)
        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]


    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    compute_avg_return(eval_env, random_policy, num_eval_episodes)

    # Please also see the metrics module for standard implementations of different
    # metrics.

    ## Data Collection

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)

    def collect_step(environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

    for _ in range(initial_collect_steps):
        collect_step(train_env, random_policy)

    # This loop is so common in RL, that we provide standard implementations of
    # these. For more details see the drivers module.

    # Dataset generates trajectories with shape [BxTx...] where
    # T = n_step_update + 1.
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size,
        num_steps=n_step_update + 1).prefetch(3)

    iterator = iter(dataset)

    ## Training the agent

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience)

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
            returns.append(avg_return)

    steps = range(0, num_iterations + 1, eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    # plt.ylim(top=100)

