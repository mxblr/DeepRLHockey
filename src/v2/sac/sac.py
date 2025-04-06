import math
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
        return (
            torch.Tensor(self.obs1_buf[random_indices]),
            torch.Tensor(self.acts_buf[random_indices]),
            torch.Tensor(self.rews_buf[random_indices]),
            torch.Tensor(self.obs2_buf[random_indices]),
            torch.Tensor(self.done_buf[random_indices]),
        )


class ValueFunctionConfig:
    def __init__(
        self,
        input_dim: int = 1,
        hidden_layers: typing.List[int] = None,
        activation_function=nn.ReLU,
        output_activation_function=None,
        weight_initializer=nn.init.xavier_uniform_,
        bias_initializer=nn.init.zeros_,
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [256, 256]
        self.activation_function = activation_function
        self.output_activation_function = output_activation_function
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer


class ValueFunction(nn.Module):
    def __init__(self, config: ValueFunctionConfig):
        super().__init__()

        self.config = config
        layer_modules = []
        inpt_size = self.config.input_dim
        for hidden_dim in self.config.hidden_layers:
            layer_modules.append(self._get_layer(input_dim=inpt_size, output_dim=hidden_dim))
            inpt_size = hidden_dim

        layer_modules.append(self._get_layer(input_dim=inpt_size, output_dim=1))
        if self.config.output_activation_function is not None:
            layer_modules.append(self.config.output_activation_function)

        self.layers = nn.ModuleList(layer_modules)

    def _get_layer(self, input_dim, output_dim):
        layer = nn.Linear(in_features=input_dim, out_features=output_dim)
        if self.config.weight_initializer is not None:
            self.config.weight_initializer(layer.weight)
        if self.config.bias_initializer:
            self.config.bias_initializer(layer.bias)
        return layer

    def forward(self, inpt):
        output = inpt
        for layer in self.layers:
            output = layer(output)

        return output


class NormalPolicyFunctionConfig:
    def __init__(
        self,
        input_dim: int = 1,
        hidden_layers: typing.List[int] = None,
        activation_function=nn.ReLU,
        output_activation_function_mu=torch.tanh,
        output_activation_function_log_std=torch.tanh,
        weight_initializer=nn.init.xavier_uniform_,
        bias_initializer=nn.init.zeros_,
        output_dim: int = 1,
        log_std_max: int = 2,
        log_std_min: int = -20,
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [256, 256]
        self.activation_function = activation_function
        self.output_activation_function_mu = output_activation_function_mu
        self.output_activation_function_log_std = output_activation_function_log_std
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.output_dim = output_dim
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min


class MuEstimator(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fct, weight_initializer, bias_initializer):
        super(MuEstimator, self).__init__()
        layer = nn.Linear(in_features=input_dim, out_features=output_dim)
        if weight_initializer is not None:
            weight_initializer(layer.weight)
        if bias_initializer:
            bias_initializer(layer.bias)

        self.head = layer
        self.activation_fct = activation_fct

    def forward(self, inpt):
        mu = self.head(inpt)
        mu_activation_fct = self.activation_fct(mu)
        return mu, mu_activation_fct


class StdEstimator(nn.Module):
    def __init__(
        self, input_dim, output_dim, activation_fct, weight_initializer, bias_initializer, log_std_min, log_std_max
    ):
        super(StdEstimator, self).__init__()
        layer = nn.Linear(in_features=input_dim, out_features=output_dim)
        if weight_initializer is not None:
            weight_initializer(layer.weight)
        if bias_initializer:
            bias_initializer(layer.bias)

        self.head = layer
        self.activation_fct = activation_fct
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

    def forward(self, inpt):
        log_std = self.head(inpt)
        if self.activation_fct != torch.tanh:
            log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        else:
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        std = torch.exp(log_std)
        return log_std, std


class NormalPolicyFunction(nn.Module):
    def __init__(self, config: NormalPolicyFunctionConfig):
        super().__init__()

        self.config = config
        layer_modules = []
        inpt_size = self.config.input_dim
        for hidden_dim in self.config.hidden_layers:
            layer_modules.append(self._get_layer(input_dim=inpt_size, output_dim=hidden_dim))
            inpt_size = hidden_dim

        self.layers = nn.ModuleList(layer_modules)
        self.mu_estimator = MuEstimator(
            input_dim=inpt_size,
            output_dim=self.config.output_dim,
            activation_fct=self.config.output_activation_function_mu,
            weight_initializer=self.config.weight_initializer,
            bias_initializer=self.config.bias_initializer,
        )

        self.std_estimator = StdEstimator(
            input_dim=inpt_size,
            output_dim=self.config.output_dim,
            activation_fct=self.config.output_activation_function_log_std,
            weight_initializer=self.config.weight_initializer,
            bias_initializer=self.config.bias_initializer,
            log_std_max=self.config.log_std_max,
            log_std_min=self.config.log_std_min,
        )

    def _get_layer(self, input_dim, output_dim):
        layer = nn.Linear(in_features=input_dim, out_features=output_dim)
        if self.config.weight_initializer is not None:
            self.config.weight_initializer(layer.weight)
        if self.config.bias_initializer:
            self.config.bias_initializer(layer.bias)
        return layer

    def forward(self, inpt):
        output = inpt
        for layer in self.layers:
            output = layer(output)

        mu, mu_activation_fct = self.mu_estimator(output)
        log_std, std = self.std_estimator(output)

        # The equivalent of tfp.distributions.MultivariateNormalDiag is Independent(Normal(loc, diag), 1), see:
        # https://github.com/pytorch/pytorch/pull/11178#issuecomment-417902463
        normal_dist = torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)
        sample = normal_dist.sample()

        action = torch.tanh(sample)
        log_prob = self.gaussian_likelihood(sample=sample, mu=mu, log_std=log_std, std=std)
        _, _, log_prob = self.squashing_function(mu=mu, pi=sample, logp_pi=log_prob)
        return action, log_prob, mu_activation_fct, log_std, mu

    @staticmethod
    def gaussian_likelihood(sample, mu, log_std, std):
        eps = 1e-6
        term_1 = torch.pow((sample - mu) / (std + eps), 2)
        term_2 = 2 * log_std * torch.log(2 * torch.Tensor([math.pi]))
        pre_sum = -0.5 * (term_1 + term_2)
        return torch.sum(pre_sum, dim=1)

    @staticmethod
    def squashing_function(mu, pi, logp_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        eps = 1e-6
        logp_pi = logp_pi - torch.sum(torch.log(torch.clamp(1 - torch.pow(pi, 2), 0, 1) + eps), dim=1)
        return mu, pi, logp_pi


class SoftActorCriticConfig:
    """Configuration File for the Soft Actor Critic Algorithm"""

    def __init__(
        self,
        tau: float = 0.005,
        learning_rate_v: float = 3e-4,
        learning_rate_q: float = 3e-4,
        learning_rate_pi: float = 3e-4,
        learning_rate_alpha: float = 3e-4,
        discount: float = 0.99,
        target_update=1,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        initial_alpha: float = 1.0,
        dim_act: int = 3,
        dim_obs: int = 16,
        alpha: typing.Union[str, float] = "auto",
        target_entropy: typing.Union[str, float] = "auto",
        q_fct_config: ValueFunctionConfig = None,
        v_fct_config: ValueFunctionConfig = None,
        pi_fct_config: NormalPolicyFunctionConfig = None,
    ):
        self.tau = tau
        self.learning_rate_v = learning_rate_v
        self.learning_rate_q = learning_rate_q
        self.learning_rate_pi = learning_rate_pi
        self.lambda_Alpha = learning_rate_alpha
        self.discount = discount
        self.target_update = target_update
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.initial_alpha = initial_alpha
        self.dim_act = dim_act
        self.dim_obs = dim_obs
        self.alpha = alpha
        self.target_entropy = target_entropy
        self.q_fct_config = q_fct_config or ValueFunctionConfig()
        self.v_fct_config = v_fct_config or ValueFunctionConfig()
        self.pi_fct_config = pi_fct_config or NormalPolicyFunctionConfig()


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
        value_function: nn.Module,
        policy_function: nn.Module,
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
        super().__init__()

        self.input_space = input_space
        self.action_space = action_space
        self.config = config

        # Using OpenAIs buffer, because it lead to speed improvement
        self.buffer = ReplayBuffer(
            obs_dim=self.config.dim_obs,
            act_dim=self.config.dim_act,
            size=self.config.buffer_size,
        )

        # set up the value, policy and q-function networks
        self.Q1 = value_function(self.config.q_fct_config)
        self.Q2 = value_function(self.config.q_fct_config)
        self.Policy = policy_function(self.config.pi_fct_config)
        self.V = value_function(self.config.v_fct_config)
        self.V_target = value_function(self.config.v_fct_config)

        # Alpha can also be trained
        self.target_entropy = self.config.target_entropy
        if self.target_entropy == "auto":
            self.target_entropy = -self.config.dim_act
        self.target_entropy = float(self.target_entropy)  # .cast(np.float32)

        # If you want to learn alpha it has to be set to 'auto'
        self.train_alpha = False
        self.alpha = self.config.alpha
        if self.alpha == "auto":
            self.train_alpha = True
            self.log_alpha = nn.Parameter(torch.tensor(float(np.log(self.config.initial_alpha)), dtype=torch.float32))
            self.alpha = torch.exp(self.log_alpha)

        self.env = env

        # self._init_update_target_V()  # TODO

        # self._sess.run(self._update_target_V_ops_hard)  # TODO

    def load_weights(self, filepath):
        # TODO
        raise NotImplementedError("Loading model weights is not implemented yet")

    def save_weights(self, filepath):
        # TODO
        raise NotImplementedError("Saving model is not implemented yet.")

    def action(self, observation):
        """
        Returns a actions sampled from the Multivariate Normal defined by the Policy network
        observation:        observation for the agent in the form [[obs1, obs2, obs3,..., obs_n]]
        """
        sampled_action, _sampled_action_log_prob, _greedy_action, _, _ = self.Policy(observation)
        return sampled_action[0]

    def act_greedy(self, observation):
        """
        Returns the mean of the Multivariate Normal defined by the policy network - and therefore the greedy action.
        observation:        observation for the agent in the form [[obs1, obs2, obs3,..., obs_n]]
        """
        _sampled_action, _sampled_action_log_prob, greedy_action, _, _ = self.Policy(observation)
        return greedy_action[0]

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
        """Calculate losses for a given set of observations, actions and rewards."""
        q1 = self.Q1(torch.cat((observation, action), dim=-1))
        q2 = self.Q2(torch.cat((observation, action), dim=-1))
        pi_action, pi_log_prob, _, pi_log_std, pi_mu = self.Policy(observation)

        # calculate value of current and new observation
        v = self.V(observation)
        v_target = self.V(observation_new)

        # Q function values for new actions
        with torch.no_grad():
            q1_pi = self.Q1(torch.cat((observation, pi_action), dim=-1))
            q2_pi = self.Q2(torch.cat((observation, pi_action), dim=-1))

        # probability of actions sampled from Multivariate normal
        log_prob_new_act = pi_log_prob
        # Uniform normal assumed - therefore prior is 0
        log_prob_prior = 0.0

        # Target for Q function
        target_q = reward + self.config.discount * (1 - env_done) * v_target.detach()
        q1_loss = 0.5 * torch.mean(torch.pow(target_q - q1, 2))
        q2_loss = 0.5 * torch.mean(torch.pow(target_q - q2, 2))

        # Target for value network
        min_q1_q2 = torch.minimum(q1_pi.detach(), q2_pi.detach())
        target_v = min_q1_q2.detach() - self.alpha.detach() * (log_prob_new_act.detach() + log_prob_prior)
        v_loss = 0.5 * torch.mean(torch.pow(v - target_v, 2))

        # PI update
        pi_loss_kl = torch.mean(self.alpha * log_prob_new_act - q1_pi)
        policy_regularization_loss = (
            0.001 * 0.5 * (torch.mean(torch.pow(pi_log_std, 2)) + torch.mean(torch.pow(pi_mu, 2)))
        )
        pi_loss = policy_regularization_loss + pi_loss_kl

        losses = [v_loss, q1_loss, q2_loss, pi_loss]
        if self.train_alpha:
            alpha_loss = -torch.mean(self.log_alpha * log_prob_new_act.detach() + self.target_entropy)
            losses.append(alpha_loss)
        return losses

    def train_from_buffer(self):
        observation, action, reward, observation_new, env_done = self.buffer.sample_batch(self.config.batch_size)
        return self.forward(observation, action, reward, observation_new, env_done)

    @staticmethod
    def do_optimizer_step(optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()

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
        bar = progressbar.ProgressBar(maxval=epochs)
        bar.start()

        # Initialize training statistics
        history = TrainingHistory()
        optimizers = {
            "q1": torch.optim.Adam(self.Q1.parameters(), lr=self.config.learning_rate_q),
            "q2": torch.optim.Adam(self.Q2.parameters(), lr=self.config.learning_rate_q),
            "v": torch.optim.Adam(self.V.parameters(), lr=self.config.learning_rate_v),
            "pi": torch.optim.Adam(self.Policy.parameters(), lr=self.config.learning_rate_pi),
        }

        # start training
        total_steps = 0
        env_done = False
        for epoch in range(epochs):
            # reset the environment
            ob, _info = self.env.reset()

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
                        a = self.action(torch.Tensor(ob).view(1, self.config.dim_obs))

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
                        loss_v, loss_q1, loss_q2, loss_pi, *_ = self.train_from_buffer()
                        self.do_optimizer_step(optimizers["q1"], loss_q1)
                        self.do_optimizer_step(optimizers["q2"], loss_q2)
                        self.do_optimizer_step(optimizers["v"], loss_v)
                        self.do_optimizer_step(optimizers["pi"], loss_pi)

                        history.update(
                            loss_pi=float(loss_pi.detach().numpy()),
                            loss_q1=float(loss_q1.detach().numpy()),
                            loss_q2=float(loss_q2.detach().numpy()),
                            loss_v=float(loss_v.detach().numpy()),
                        )
                total_steps += 1
                if env_done:
                    break

            history.update(episode_reward=total_reward)
            if epoch % n_log_epochs == 0:
                history.plot()
            bar.update(epoch)

        return history.episode_rewards
