import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append(".")
from src.Environment import *
from src.param import *
# from src.Data.Random import *
from src.TorchVersion.pytorch_op import *
from src.TorchVersion.pytorch_gcn import *
import time


class Actor_Agent(nn.Module):
    def __init__(self, policy_input_dim=args.policy_input_dim,
                 hid_dims=args.hid_dims,
                 output_dim=args.output_dim,
                 max_depth=args.max_depth, eps=1e-6):
        super(Actor_Agent, self).__init__()
        self.policy_input_dim = policy_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth

        self.input_layer = torch.nn.Linear(self.policy_input_dim + self.output_dim, 32)
        self.hiden_layer1 = torch.nn.Linear(32, 16)
        self.hiden_layer2 = torch.nn.Linear(16, 8)
        self.output_layer = torch.nn.Linear(8, 1)
        self.act_fn = leaky_relu

        self.gcn_layers = GraphCNN(self.policy_input_dim,
                                   self.hid_dims,
                                   self.output_dim,
                                   self.max_depth,
                                   self.act_fn)

        # self.env = None

    def forward(self, input, reachable_edge_matrix):
        gcn_outputs = self.gcn_layers(input, reachable_edge_matrix)
        # print(input.shape, gcn_outputs.shape)
        input = torch.cat((input, gcn_outputs), 1)
        # print(input.shape)

        policy_outputs = self.act_fn(self.input_layer(input))
        policy_outputs = self.act_fn(self.hiden_layer1(policy_outputs))
        policy_outputs = self.act_fn(self.hiden_layer2(policy_outputs))

        policy_outputs = self.output_layer(policy_outputs)
        policy_outputs = torch.Tensor.reshape(policy_outputs, [1, -1])

        policy_min = torch.min(policy_outputs)
        # print(policy_outputs, policy_min)
        policy_outputs = torch.subtract(policy_outputs, policy_min)
        # print(policy_outputs)
        policy_max = torch.max(policy_outputs)
        # print(policy_outputs, policy_max)
        policy_outputs = torch.divide(policy_outputs, policy_max)
        # print(policy_outputs)
        policy_outputs = torch.nn.Softmax(dim=-1)(policy_outputs)
        # print(policy_outputs, torch.sum(policy_outputs))
        return policy_outputs



