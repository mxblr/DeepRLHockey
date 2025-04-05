import typing

import numpy as np
import progressbar
import torch
import torch.nn as nn

from src.v2.sac.training_utils import TrainingHistory


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int):
        """Set up buffers"""
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs: float, act: float, rew: float, next_obs: float, done: float):
        """Store environment state in buffer"""
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int = 32):
        """Sample random observations from the buffer

        :param batch_size: Number of buffer entries to sample.
        :return: Dictionary of observations, action, reward and whether the environment is done."""
        random_indices = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[random_indices],
            obs2=self.obs2_buf[random_indices],
            acts=self.acts_buf[random_indices],
            rews=self.rews_buf[random_indices],
            done=self.done_buf[random_indices],
        )


class SoftActorCriticConfig:
    """Configuration File for the Soft Actor Critic Algorithm"""

    def __init__(
        self,
        tau: float = 0.005,
        lambda_V: float = 3e-4,
        lambda_Q: float = 3e-4,
        lambda_Pi: float = 3e-4,
        lambda_Alpha: float = 3e-4,
        discount: float = 0.99,
        target_update=1,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        initial_alpha: float = 1.0,
        dim_act: int = 3,
        dim_obs: int = 16,
        alpha: typing.Union[str, float] = "auto",
        target_entropy: typing.Union[str, float] = "auto",
        q_fct_config: dict = None,
        v_fct_config: dict = None,
        pi_fct_config: dict = None,
    ):
        self.tau = tau
        self.lambda_V = lambda_V
        self.lambda_Q = lambda_Q
        self.lambda_Pi = lambda_Pi
        self.lambda_Alpha = lambda_Alpha
        self.discount = discount
        self.target_update = target_update
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.initial_alpha = initial_alpha
        self.dim_act = dim_act
        self.dim_obs = dim_obs
        self.alpha = alpha
        self.target_entropy = target_entropy
        self.q_fct_config = q_fct_config
        self.v_fct_config = v_fct_config
        self.pi_fct_config = pi_fct_config


class SoftActorCritic(nn.Module):
    """
    Implementation of the Soft-Actor-Critic algorithm.

    We assume uniform priors.
    """

    def __init__(
        self,
        config: SoftActorCriticConfig,
        input_space,
        action_space,
        value_fct: nn.Module,
        policy_fct: nn.Module,
        env,
    ):
        """
        Initialization:

        :param o_space:        input space, e.g. containing information about dimensionality
        :param a_space:        actions space, e.g. containing information about dimensionality
        :param value_fct:      value function to use. Here feed forward Neural Networks are used, but others are
                                possible.
        :param policy_fct:     policy function to use. Here only Gaussian is implemented, but others are possible.
        :param env:            gym environment, necessary for the train function - if you define your own training
                                function this is not necessary
        :param q_fct_config:   configaration file for the Q-functions (e.g. Layers, Activation function,...)
        :param v_fct_config:   configaration file for the value functions (e.g. Layers, Activation function,...)
        :param pi_fct_config:  configaration file for the policy network (e.g. Layers, Activation function,...)
        :param scope:          Scope of the agent
        :param save_path:      Path to the directory where you want to save your network weights and graphs
        :param user_config:    contains additional parameters saved in self._config dictionary
        """

        self.input_space = input_space
        self.action_space = action_space
        self.config = config

        # Using OpenAIs buffer, because it lead to speed improvement
        self.buffer = ReplayBuffer(
            obs_dim=self.config.dim_obs,
            act_dim=self.configdim_act,
            size=self.config.buffer_size,
        )

        self.value_function = value_fct
        self.policy_function = policy_fct

        # Alpha can also be trained
        self.target_entropy = self.config.target_entropy
        if self.target_entropy == "auto":
            self.target_entropy = -self.config.dim_act
        self.target_entropy = self.target_entropy.astype(np.float32)

        # If you want to learn alpha it has to be set to 'auto'
        self.train_alpha = False
        self.alpha = self.config.alpha
        if self.alpha == "auto":
            self.train_alpha = True
            self.log_alpha = nn.Parameter(torch.tensor(float(np.log(self.config.initial_alpha)), dtype=torch.float32))
            self.alpha = torch.exp(self.log_alpha)

        self.env = env

        self._init_update_target_V()  # TODO

        self._sess.run(self._update_target_V_ops_hard)  # TODO

    def load_weights(self, filepath):
        pass

    def save_weights(self, filepath):
        pass

    def action(self, observation):
        """
        Returns a actions sampled from the Multivariate Normal defined by the Policy network
        observation:        observation for the agent in the form [[obs1, obs2, obs3,..., obs_n]]
        """
        actions = self.Policy.act(observation)
        return actions[0]

    def act_greedy(self, observation):
        """
        Returns the mean of the Multivariate Normal defined by the policy network - and therefore the greedy action.
        observation:        observation for the agent in the form [[obs1, obs2, obs3,..., obs_n]]
        """
        actions = self.Policy.mu_tanh(observation)
        return actions[0]

    def reverse_action(self, action):
        """
        If a environment wrapper is used you can reverse it here.
        """
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action

    def forward(self, observation, action, reward, observation_new, env_done):
        loss_v_fct, loss_q1_fct, loss_q2_fct, loss_pi_fct = None, None, None, None
        return loss_v_fct, loss_q1_fct, loss_q2_fct, loss_pi_fct

    def train_from_buffer(self):
        observation, action, reward, observation_new, env_done = self.buffer.sample_batch(self.config.batch_size)
        return self.forward(observation, action, reward, observation_new, env_done)

    def train(
        self,
        epochs: int = 1000,
        max_steps: int = 500,
        env_steps: int = 1,
        grad_steps: int = 1,
        n_burn_in_steps: int = 1000,
        n_log_epochs: int = 1,
    ):
        """
        Internal training method, can be overwritten for other environments

        :param epochs:       number of training steps
        :param max_steps:       number of maximal steps per episode
        :param env_steps:      number of environment steps per iter_fit step
        :param grad_steps:     number of training steps per iter_fit step
        :param n_burn_in_steps:        number of initial steps sampled from uniform distribution over actions
        :param n_log_epochs:    Logging frequency in epochs. Defaults to 1.
        """
        bar = progressbar.ProgressBar(max_value=epochs)
        # Initialize training statistics
        history = TrainingHistory()

        # start training
        total_steps = 0
        env_done = False
        for epoch in range(epochs):
            # reset the environment
            ob = self.env.reset()

            total_reward = 0
            # sample observations
            for _ in range(max_steps):
                for _episode_step in range(env_steps):
                    if total_steps < n_burn_in_steps:
                        # choose random action during the burn in phase
                        a = self.env.action_space.sample()
                        a = self.reverse_action(a)
                    else:
                        # choose an action based on our model
                        a = self.action(np.asarray(ob).reshape(1, self._o_space.shape[0]))
                        a = a[0]

                    # execute the action and receive updated observations and reward
                    ob_new, reward, env_done, *_info = self.env.step(a)
                    total_reward += reward

                    # store the action and observations
                    self.buffer.store(ob, a, reward, ob_new, env_done)
                    ob = ob_new

                # update weights
                if total_steps >= self.config.batch_size:
                    for _gradient_step in range(grad_steps):
                        # train the actor and critic models
                        loss_v_fct, loss_q1_fct, loss_q2_fct, loss_pi_fct = self.train_from_buffer()
                        history.update(loss_pi=loss_pi_fct, loss_q1=loss_q1_fct, loss_q2=loss_q2_fct, loss_v=loss_v_fct)
                total_steps += 1
                if env_done:
                    break

            history.update(episode_reward=total_reward)
            if epoch % n_log_epochs == 0:
                history.plot()
            bar.update(epoch)

        return history.episode_rewards
