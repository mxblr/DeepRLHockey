from unittest import TestCase

import torch
from torch import nn

from src.v2.sac.sac import NormalPolicyFunction, NormalPolicyFunctionConfig


class TestNormalPolicy(TestCase):
    def test_setup(self):
        config = NormalPolicyFunctionConfig()
        policy_network = NormalPolicyFunction(config)
        self.assertIsInstance(policy_network, nn.Module)

    def test_forward_pass(self):
        config = NormalPolicyFunctionConfig()
        policy_network = NormalPolicyFunction(config)
        action, *other_outputs = policy_network(torch.rand((1, config.input_dim)))
        self.assertIsInstance(action, torch.Tensor)
        self.assertEqual(action.shape, (1, config.output_dim))

        for out in other_outputs:
            self.assertIsInstance(out, torch.Tensor)

    def test_overfitting(self):
        config = NormalPolicyFunctionConfig()
        policy_network = NormalPolicyFunction(config)
        inpt = torch.rand((256, config.input_dim))
        q1_pi = torch.rand((256, config.input_dim))
        optimizer = torch.optim.Adam(policy_network.parameters(), lr=3e-4)
        _alpha_no_grad = 0.1

        for step in range(100):
            pi_action, log_prob_new_act, _, pi_log_std, pi_mu = policy_network(inpt)
            pi_loss_kl = torch.mean(_alpha_no_grad * log_prob_new_act - q1_pi)
            policy_regularization_loss = (
                0.001 * 0.5 * (torch.mean(torch.pow(pi_log_std, 2)) + torch.mean(torch.pow(pi_mu, 2)))
            )
            pi_loss = pi_loss_kl + policy_regularization_loss
            print(f"{step}: {pi_loss=} | {pi_loss_kl=} | {policy_regularization_loss=}")
            optimizer.zero_grad()
            pi_loss.backward()
            optimizer.step()
        self.assertLess(pi_loss.detach().numpy(), 0.001)
