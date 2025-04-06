from unittest import TestCase

import torch
from torch import nn

from src.v2.sac.sac import ValueFunctionConfig, ValueFunction


class TestValueFunction(TestCase):
    def test_setup(self):
        config = ValueFunctionConfig()
        policy_network = ValueFunction(config)
        self.assertIsInstance(policy_network, nn.Module)

    def test_forward_pass(self):
        config = ValueFunctionConfig()
        policy_network = ValueFunction(config)
        out = policy_network(torch.rand((1, config.input_dim)))
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (1, 1))

