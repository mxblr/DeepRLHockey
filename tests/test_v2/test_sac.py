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


class TestSAC(TestCase):
    def setUp(self) -> None:
        env = gym.make("Pendulum-v1")

        self.config = SoftActorCriticConfig(
            q_fct_config=ValueFunctionConfig(),
            v_fct_config=ValueFunctionConfig(),
            pi_fct_config=NormalPolicyFunctionConfig(),
            batch_size=4,
        )

        self.sac = SoftActorCritic(
            self.config,
            policy_function=NormalPolicyFunction,
            value_function=ValueFunction,
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
        action = self.sac.forward(torch.rand((1, 3)))
        self.assertIsInstance(action, np.ndarray)

    def test_greedy_action(self):
        action = self.sac.act_greedy(torch.rand((1, 3)))
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
