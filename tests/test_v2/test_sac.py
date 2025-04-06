from unittest import TestCase

import gymnasium as gym
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
        ac_space = env.action_space
        o_space = env.observation_space

        self.config = SoftActorCriticConfig(
            q_fct_config=ValueFunctionConfig(input_dim=4),
            v_fct_config=ValueFunctionConfig(input_dim=3),
            pi_fct_config=NormalPolicyFunctionConfig(input_dim=3),
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

    def test_forward_pass(self):
        ob, _info = self.sac.env.reset()
        a = self.sac.env.action_space.sample()
        ob_new, reward, env_done, *_info = self.sac.env.step(a)

        losses = self.sac.forward(
            observation=torch.Tensor([ob]),
            action=torch.Tensor([a]),
            reward=torch.Tensor([reward]),
            observation_new=torch.Tensor([ob_new]),
            env_done=torch.Tensor([env_done]),
        )
        self.assertEqual(len(losses), 5)
        for loss in losses:
            self.assertIsInstance(loss, torch.Tensor)
            self.assertTrue(loss.requires_grad)

    def test_action(self):
        action = self.sac.action(torch.rand((1, self.config.pi_fct_config.input_dim)))
        self.assertIsInstance(action, torch.Tensor)

    def test_greedy_action(self):
        action = self.sac.act_greedy(torch.rand((1, self.config.pi_fct_config.input_dim)))
        self.assertIsInstance(action, torch.Tensor)

    def test_reverse_action(self):
        a = self.sac.env.action_space.sample()
        a_rev = self.sac.reverse_action(a)
        self.assertIsInstance(a_rev, type(a))
