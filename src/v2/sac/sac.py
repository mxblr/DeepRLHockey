import itertools
import typing
from copy import deepcopy

import gymnasium
import numpy as np
import progressbar
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.v2.sac.training_utils import NormalizedActions, TrainingHistory


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
        hidden_layers: typing.List[int] = None,
        activation_function=nn.ReLU,
        output_activation_function=None,
        weight_initializer=nn.init.xavier_uniform_,
        bias_initializer=nn.init.zeros_,
    ):
        self.hidden_layers = hidden_layers or [256, 256]
        self.activation_function = activation_function
        self.output_activation_function = output_activation_function
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer


class ValueFunction(nn.Module):
    def __init__(self, config: ValueFunctionConfig, input_dim):
        super().__init__()

        self.config = config
        layer_modules = []
        inpt_size = input_dim
        for hidden_dim in self.config.hidden_layers:
            layer_modules.append(self._get_layer(input_dim=inpt_size, output_dim=hidden_dim))
            layer_modules.append(self.config.activation_function())
            inpt_size = hidden_dim

        layer_modules.append(self._get_layer(input_dim=inpt_size, output_dim=1))
        if self.config.output_activation_function is not None:
            layer_modules.append(self.config.output_activation_function)

        self.layers = nn.Sequential(*layer_modules)

    def _get_layer(self, input_dim, output_dim):
        layer = nn.Linear(in_features=input_dim, out_features=output_dim)
        if self.config.weight_initializer is not None:
            self.config.weight_initializer(layer.weight)
        if self.config.bias_initializer:
            self.config.bias_initializer(layer.bias)
        return layer

    def forward(self, inpt):
        return self.layers(inpt)


class NormalPolicyFunctionConfig:
    def __init__(
        self,
        hidden_layers: typing.List[int] = None,
        activation_function=nn.ReLU,
        output_activation_function_mu=torch.tanh,
        output_activation_function_log_std=torch.tanh,
        weight_initializer=nn.init.xavier_uniform_,
        bias_initializer=nn.init.zeros_,
        log_std_max: int = 2,
        log_std_min: int = -20,
    ):
        self.hidden_layers = hidden_layers or [256, 256]
        self.activation_function = activation_function
        self.output_activation_function_mu = output_activation_function_mu
        self.output_activation_function_log_std = output_activation_function_log_std
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
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
        if self.activation_fct:
            log_std = self.activation_fct(log_std)

        if self.activation_fct != torch.tanh:
            log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        else:
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        std = torch.exp(log_std)
        return log_std, std


class NormalPolicyFunction(nn.Module):
    def __init__(self, config: NormalPolicyFunctionConfig, input_dim: int, output_dim: int):
        super().__init__()

        self.config = config
        layer_modules = []
        inpt_size = input_dim
        for hidden_dim in self.config.hidden_layers:
            layer_modules.append(self._get_layer(input_dim=inpt_size, output_dim=hidden_dim))
            layer_modules.append(self.config.activation_function())
            inpt_size = hidden_dim

        self.layers = nn.Sequential(*layer_modules)
        self.mu_estimator = MuEstimator(
            input_dim=inpt_size,
            output_dim=output_dim,
            activation_fct=self.config.output_activation_function_mu,
            weight_initializer=self.config.weight_initializer,
            bias_initializer=self.config.bias_initializer,
        )

        self.std_estimator = StdEstimator(
            input_dim=inpt_size,
            output_dim=output_dim,
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
        output = self.layers(inpt)

        mu, mu_activation_fct = self.mu_estimator(output)
        log_std, std = self.std_estimator(output)

        # The equivalent of tfp.distributions.MultivariateNormalDiag is Independent(Normal(loc, diag), 1), see:
        # https://github.com/pytorch/pytorch/pull/11178#issuecomment-417902463
        normal_dist = torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)

        # sample an action from the distribution - using the re-parametrization trick (rsample) to make sure we can
        # propagate the training signal back through the samplig
        sample = normal_dist.rsample()

        log_prob = normal_dist.log_prob(sample)  # .sum(dim=-1)
        log_prob -= 2 * (np.log(2) - sample - F.softplus(-2 * sample)).sum(dim=1)

        action = torch.tanh(sample)
        return action, log_prob, mu_activation_fct, log_std, mu


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
        alpha: typing.Union[str, float] = "auto",
        target_entropy: typing.Union[str, float] = "auto",
        q_fct_config: ValueFunctionConfig = None,
        v_fct_config: ValueFunctionConfig = None,
        pi_fct_config: NormalPolicyFunctionConfig = None,
        max_grad_norm: float = None,
        normalize_actions: bool = True,
    ):
        self.tau = tau
        self.learning_rate_v = learning_rate_v
        self.learning_rate_q = learning_rate_q
        self.learning_rate_pi = learning_rate_pi
        self.learning_rate_alpha = learning_rate_alpha
        self.discount = discount
        self.target_update = target_update
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.initial_alpha = initial_alpha
        self.alpha = alpha
        self.target_entropy = target_entropy
        self.q_fct_config = q_fct_config or ValueFunctionConfig()
        self.v_fct_config = v_fct_config or ValueFunctionConfig()
        self.pi_fct_config = pi_fct_config or NormalPolicyFunctionConfig()
        self.max_grad_norm = max_grad_norm
        self.normalize_actions = normalize_actions


class SoftActorCritic(nn.Module):
    """
    Implementation of the Soft-Actor-Critic algorithm.

    We assume uniform priors.
    """

    def __init__(
        self,
        config: SoftActorCriticConfig,
        value_function: typing.Type[ValueFunction],
        policy_function: typing.Type[NormalPolicyFunction],
        env,
    ):
        """
        Initialization:
        :param config           Configuration file for the SAC Agent
        :param value_function:  value function to use. Here feed forward Neural Networks are used, but others are
                                possible.
        :param policy_function: policy function to use. Here only Gaussian is implemented, but others are possible.
        :param env:             gym environment, necessary for the train function - if you define your own training
                                function this is not necessary
        """
        super().__init__()
        self.config = config

        # normalize the actions into range -1 to 1, to be compatible with tanh activated actions
        self.env = NormalizedActions(env) if self.config.normalize_actions else env
        self.input_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.dim_obs, *_ = self.input_space.shape
        self.dim_act, *_ = self.action_space.shape

        # Using OpenAIs buffer, because it lead to speed improvement
        self.buffer = ReplayBuffer(obs_dim=self.dim_obs, act_dim=self.dim_act, size=self.config.buffer_size)

        # set up the value, policy and q-function networks
        self.Q1 = value_function(self.config.q_fct_config, input_dim=self.dim_obs + self.dim_act)
        self.Q2 = value_function(self.config.q_fct_config, input_dim=self.dim_obs + self.dim_act)
        self.Policy = policy_function(self.config.pi_fct_config, input_dim=self.dim_obs, output_dim=self.dim_act)
        self.V = value_function(self.config.v_fct_config, input_dim=self.dim_obs)
        self.V_target = deepcopy(self.V)
        # we will not update the V_target network, hence we can freeze the parameters
        for p in self.V_target.parameters():
            p.requires_grad = False

        # Alpha can be trained
        self.target_entropy = self.config.target_entropy
        if self.target_entropy == "auto":
            self.target_entropy = -self.dim_act
        self.target_entropy = float(self.target_entropy)

        # If you want to learn alpha it has to be set to 'auto'
        self.train_alpha = False
        self.alpha = self.config.alpha
        if self.alpha == "auto":
            self.train_alpha = True
            self.log_alpha = nn.Parameter(torch.tensor(float(np.log(self.config.initial_alpha)), dtype=torch.float32))
            self.alpha = torch.exp(self.log_alpha)

    def load_weights(self, filepath):
        # TODO
        raise NotImplementedError("Loading model weights is not implemented yet")

    def save_weights(self, filepath):
        # TODO
        raise NotImplementedError("Saving model is not implemented yet.")

    def forward(self, observation: torch.Tensor) -> np.ndarray:
        """Returns a actions sampled from the Multivariate Normal defined by the Policy network

        :param observation: Observation for the agent in the form [obs1]
        :return:            Action sampled from the policy network, as a np.array
        """
        with torch.no_grad():
            sampled_action, _sampled_action_log_prob, _greedy_action, _, _ = self.Policy(observation)
        return sampled_action.numpy()[0]

    def act(self, observation: torch.Tensor) -> np.ndarray:
        """Returns a actions sampled from the Multivariate Normal defined by the Policy network

        Same as "forward".

        :param observation: Observation for the agent in the form [obs1]
        :return:            Action sampled from the policy network, as a np.array
        """
        return self.forward(torch.as_tensor(observation).view(1, self.dim_obs).float())

    def act_greedy(self, observation: torch.Tensor) -> np.ndarray:
        """Returns the mean of the Multivariate Normal defined by the policy network - and therefore the greedy action.

        :param observation:        observation for the agent in the form [obs1]
        :return:                   Action, as the mean of the Normal from the policy network, as a np.array
        """

        with torch.no_grad():
            _sampled_action, _sampled_action_log_prob, greedy_action, _, _ = self.Policy(
                torch.as_tensor(observation).view(1, self.dim_obs).float()
            )
        return greedy_action.numpy()[0]

    def get_q_losses(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        observation_new: torch.Tensor,
        env_done: torch.Tensor,
    ) -> typing.Dict[str, torch.Tensor]:
        """Calculate losses for Q networks

        :param observation: observation received from the environment as a torch tensor (b, dim_observation)
        :param action: action received from the environment as a torch tensor (b, dim_action)
        :param reward: reward received from the environment as a torch tensor (b, dim_reward)
        :param observation_new: observation_new received from the environment as a torch tensor (b, dim_observation_new)
        :param env_done: env_done received from the environment as a torch tensor (b, dim_env_done)
        :return: Dictionary containing losses for Q1 and Q2 (under keys "q1" and "q2" respectively).
        """
        # Get Q values for current observation and the action that was taken
        q1 = self.Q1(torch.cat((observation, action), dim=-1))
        q2 = self.Q2(torch.cat((observation, action), dim=-1))

        with torch.no_grad():
            # Get the estimated value of the new environment state / observation
            v_target_val = self.V_target(observation_new)
            # calculate target for q networks as reward + estimated value for new state (if env is not done=
            target_q = reward.view(-1, 1) + (self.config.discount * (1 - env_done)).view(-1, 1) * v_target_val

        # Target for Q functions
        q1_loss = 0.5 * torch.mean(torch.pow(target_q - q1, 2))
        q2_loss = 0.5 * torch.mean(torch.pow(target_q - q2, 2))
        return {"q1": q1_loss, "q2": q2_loss}

    def get_v_pi_alpha_losses(self, observation: torch.Tensor) -> typing.Dict[str, torch.Tensor]:
        """Calculate losses for policy, value networks and alpha

        :param observation: observation received from the environment as a torch tensor (b, dim_observation)
        :return: Dictionary containing losses for Value and Policy networks and alpha
                 (under keys "v", "pi" and "alpha" respectively).
        """
        # sample an action from the policy network for the current observation
        pi_action, log_prob_new_act, _, pi_log_std, pi_mu = self.Policy(observation)

        # get an estimate for the Q values for the action
        q1_pi = self.Q1(torch.cat((observation, pi_action), dim=-1))
        q2_pi = self.Q2(torch.cat((observation, pi_action), dim=-1))

        # ------------------------------- Calculate Policy loss -------------------------------
        _alpha_no_grad = self.alpha.detach() if self.train_alpha else self.alpha
        pi_loss_kl = torch.mean((_alpha_no_grad * log_prob_new_act).view(-1, 1) - q1_pi)
        policy_regularization_loss = (
            0.001 * 0.5 * (torch.mean(torch.pow(pi_log_std, 2)) + torch.mean(torch.pow(pi_mu, 2)))
        )
        pi_loss = pi_loss_kl + policy_regularization_loss

        # ------------------------------- Calculate Value loss -------------------------------
        # Uniform normal assumed - therefore prior is 0
        log_prob_prior = 0.0
        min_q1_q2 = torch.minimum(q1_pi.detach(), q2_pi.detach())
        target_v = min_q1_q2.detach() - _alpha_no_grad * (log_prob_new_act.detach().view(-1, 1) + log_prob_prior)

        v = self.V(observation)
        v_loss = 0.5 * torch.mean(torch.pow(v - target_v, 2))

        losses = {"v": v_loss, "pi": pi_loss}
        # ------------------------------- Calculate Alpha loss -------------------------------
        alpha_loss = None
        if self.train_alpha:
            alpha_loss = -torch.mean(self.log_alpha * (log_prob_new_act.detach() + self.target_entropy).detach())
        losses["alpha"] = alpha_loss
        return losses

    def update_v_target(self):
        """Apply polyak-averaging to update the weights of V_target"""
        target_multiplier = 1 - self.config.tau
        source_multiplier = self.config.tau

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for v, v_target in zip(self.V.parameters(), self.V_target.parameters(), strict=False):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                v_target.data.mul_(target_multiplier)
                v_target.data.add_(source_multiplier * v.data)

    def train(
        self,
        epochs: int = 1000,
        max_steps: int = 500,
        env_steps: int = 1,
        grad_steps: int = 1,
        n_burn_in_steps: int = 1000,
        n_log_epochs: int = 1,
        log_output: str = "plot",
        opponent_agent=None,
    ) -> float:
        """
        Internal training method, can be overwritten for other environments

        :param epochs:       number of training steps
        :param max_steps:       number of maximal steps per episode
        :param env_steps:      number of environment steps per iter_fit step
        :param grad_steps:     number of training steps per iter_fit step
        :param n_burn_in_steps:        number of initial steps sampled from uniform distribution over actions
        :param n_log_epochs:    Logging frequency in epochs. Defaults to 1.
        :param log_output: Where to log to based on the TrainingHistory class.
        :param opponent_agent: If playing a game with two opponents, this is the opponent
        :return: Total reward accumulated
        """
        bar = progressbar.ProgressBar(maxval=epochs)
        bar.start()

        # Initialize training statistics
        history = TrainingHistory(log_target=log_output)
        optimizers = {
            "q1": torch.optim.Adam(self.Q1.parameters(), lr=self.config.learning_rate_q),
            "q2": torch.optim.Adam(self.Q2.parameters(), lr=self.config.learning_rate_q),
            "v": torch.optim.Adam(self.V.parameters(), lr=self.config.learning_rate_v),
            "pi": torch.optim.Adam(self.Policy.parameters(), lr=self.config.learning_rate_pi),
        }
        if self.train_alpha:
            optimizers["alpha"] = torch.optim.Adam([self.log_alpha], lr=self.config.learning_rate_alpha)

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
                    ob, env_done, episode_reward = self.fill_buffer(
                        ob, total_steps=total_steps, n_burn_in_steps=n_burn_in_steps, opponent_agent=opponent_agent
                    )
                    total_reward += episode_reward
                    if env_done:
                        break

                # update weights
                if self.buffer.size >= self.config.batch_size:
                    for _gradient_step in range(grad_steps):
                        # train the actor and critic models
                        batch_observation, batch_action, batch_reward, batch_observation_new, batch_env_done = (
                            self.buffer.sample_batch(self.config.batch_size)
                        )
                        q_losses = self.get_q_losses(
                            batch_observation, batch_action, batch_reward, batch_observation_new, batch_env_done
                        )

                        optimizers["q1"].zero_grad()
                        q_losses["q1"].backward()

                        optimizers["q2"].zero_grad()
                        q_losses["q2"].backward()

                        optimizers["q1"].step()
                        optimizers["q2"].step()

                        # optimize value and policy networks
                        q_params = itertools.chain(self.Q1.parameters(), self.Q2.parameters())
                        for param in q_params:
                            param.requires_grad = False

                        v_pi_alpha_losses = self.get_v_pi_alpha_losses(batch_observation)

                        optimizers["v"].zero_grad()
                        v_pi_alpha_losses["v"].backward()

                        optimizers["pi"].zero_grad()
                        v_pi_alpha_losses["pi"].backward()

                        if self.train_alpha:
                            optimizers["alpha"].zero_grad()
                            v_pi_alpha_losses["alpha"].backward()

                        if self.config.max_grad_norm is not None:
                            for optimizer in optimizers.values():
                                torch.nn.utils.clip_grad_norm_(
                                    parameters=optimizer.param_groups[0]["params"], max_norm=self.config.max_grad_norm
                                )

                        optimizers["v"].step()
                        optimizers["pi"].step()

                        if self.train_alpha:
                            optimizers["alpha"].step()

                        q_params = itertools.chain(self.Q1.parameters(), self.Q2.parameters())
                        for param in q_params:
                            param.requires_grad = True

                        self.update_v_target()
                        history.update(
                            loss_pi=float(v_pi_alpha_losses["pi"].detach().numpy()),
                            loss_q1=float(q_losses["q1"].detach().numpy()),
                            loss_q2=float(q_losses["q2"].detach().numpy()),
                            loss_v=float(v_pi_alpha_losses["v"].detach().numpy()),
                            loss_alpha=float(v_pi_alpha_losses["alpha"].detach().numpy()) if self.train_alpha else None,
                        )
                total_steps += 1
                if env_done:
                    break

            history.update(episode_reward=total_reward)
            if epoch % n_log_epochs == 0:
                history()
            bar.update(epoch)

        return history

    def fill_buffer(self, ob, total_steps: int, n_burn_in_steps: int, opponent_agent=None):
        if total_steps < n_burn_in_steps:
            # choose random action during the burn in phase
            # sampled from the normal action space
            a = self.env.action_space.sample()
            # normalize
            if self.config.normalize_actions:
                a = self.env.reverse_action(a)
        else:
            # choose an action based on our model
            a = self.act(ob)

        # if we play a game with an opponent, sample the opponent action
        store_a = a
        if opponent_agent:
            ob_opponent_agent = self.env.obs_agent_two()
            a_opponent_agent = opponent_agent.act(ob_opponent_agent)
            a = np.hstack([a, a_opponent_agent])

        # execute the action and receive updated observations and reward
        ob_new, reward, terminated, truncated, *_info = self.env.step(a)
        env_done = terminated or truncated
        # store the action and observations
        self.buffer.store(ob, store_a, float(reward), ob_new, env_done)
        return ob_new, env_done, reward

    def run_agent_on_env(
        self, env: gymnasium.Env, greedy_action: bool = True, max_steps: int = 1000, opponent_agent=None
    ) -> float:
        """Run the agent on an environment

        :param env: The gymnasium Environment
        :param greedy_action:  Whether to sample an action from the multivariate normal or to take
                               mean as the action. Defaults to True, which means greedily taking the mean.
        :param max_steps:      Maximum number of steps to take in the environment
        :return: Total reward accumulated
        """
        env = NormalizedActions(env) if self.config.normalize_actions else env
        ob, _info = env.reset()
        total_reward = 0
        for _step in range(max_steps):
            a = self.act_greedy(ob) if greedy_action else self.act(ob)
            # if we play a game with an opponent, sample the opponent action
            if opponent_agent:
                ob_opponent_agent = env.obs_agent_two()  # noqa
                a_opponent_agent = opponent_agent.act(ob_opponent_agent)
                a = np.hstack([a, a_opponent_agent])

            ob, reward, terminated, truncated, *_info = env.step(a)
            _env_done = terminated or truncated
            total_reward += reward

        env.close()
        return total_reward
