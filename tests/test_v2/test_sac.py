from unittest import TestCase

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from src.v2.sac.sac import (
    NormalPolicyFunction,
    NormalPolicyFunctionConfig,
    SoftActorCritic,
    SoftActorCriticConfig,
    ValueFunction,
    ValueFunctionConfig,
)
from src.v2.sac.training_utils import NormalizedActions


class TestSAC(TestCase):
    def setUp(self) -> None:
        env = NormalizedActions(gym.make("Pendulum-v1"))
        ac_space = env.action_space
        o_space = env.observation_space

        self.config = SoftActorCriticConfig(
            q_fct_config=ValueFunctionConfig(input_dim=4),
            v_fct_config=ValueFunctionConfig(input_dim=3),
            pi_fct_config=NormalPolicyFunctionConfig(input_dim=3),
            batch_size=4,
            dim_obs=3,
            dim_act=1,
        )

        self.sac = SoftActorCritic(
            self.config,
            policy_function=NormalPolicyFunction,
            value_function=ValueFunction,
            input_space=o_space,
            action_space=ac_space,
            env=env,
        )

    def test_setup(self):
        self.assertIsInstance(self.sac, nn.Module)

    def test_get_q_losses(self):
        ob, _info = self.sac.env.reset()
        a = self.sac.env.action_space.sample()
        ob_new, reward, env_done, *_info = self.sac.env.step(a)

        losses = self.sac.get_q_losses(
            observation=torch.Tensor([ob]),
            action=torch.Tensor([a]),
            reward=torch.Tensor([reward]),
            observation_new=torch.Tensor([ob_new]),
            env_done=torch.Tensor([env_done]),
        )
        self.assertEqual({"q1", "q2"}, set(losses.keys()))
        for loss in losses.values():
            self.assertIsInstance(loss, torch.Tensor)
            self.assertTrue(loss.requires_grad)

    def test_get_v_pi_alpha_losses(self):
        ob, _info = self.sac.env.reset()
        losses = self.sac.get_v_pi_alpha_losses(torch.Tensor([ob]))
        self.assertEqual({"v", "pi", "alpha"}, set(losses.keys()))
        for loss in losses.values():
            self.assertIsInstance(loss, torch.Tensor)
            self.assertTrue(loss.requires_grad)

    def test_action(self):
        action = self.sac.forward(torch.rand((1, self.config.pi_fct_config.input_dim)))
        self.assertIsInstance(action, np.ndarray)

    def test_greedy_action(self):
        action = self.sac.act_greedy(torch.rand((1, self.config.pi_fct_config.input_dim)))
        self.assertIsInstance(action, np.ndarray)

    def test_reverse_action(self):
        a = self.sac.env.action_space.sample()
        a_rev = self.sac.env.reverse_action(a)
        self.assertIsInstance(a_rev, type(a))

    def test_train(self):
        rewards = self.sac.train(
            epochs=5,
            max_steps=5,
            env_steps=1,
            grad_steps=1,
            n_burn_in_steps=1,
            n_log_epochs=100,
            log_output="stdout",
        )
        self.assertTrue(rewards)
