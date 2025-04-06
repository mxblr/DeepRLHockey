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
