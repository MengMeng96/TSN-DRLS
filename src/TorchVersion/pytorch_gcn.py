"""
Graph Convolutional Network

Propergate node features among neighbors
via parameterized message passing scheme
"""
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append(".")
import copy
import numpy as np
import torch
from pytorch_op import *
import torch.nn as nn
from param import *


class GraphCNN(nn.Module):
    def __init__(self, input_dim, hid_dims, output_dim,
                 max_depth, act_fn):
        super(GraphCNN, self).__init__()
        self.input_dim = input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.act_fn = act_fn

        # initialize message passing transformation parameters
        # h: x -> x'
        # hid_dims = [16,8], output_dim = 8
        self.prepare_layer1 = Prepare_Layer(self.input_dim, 16)
        self.prepare_layer2 = Prepare_Layer(16, 8)
        self.prepare_layer3 = Prepare_Layer(8, self.output_dim)

        # g: e -> e
        self.aggregate_layer1 = Prepare_Layer(self.input_dim, 16)
        self.aggregate_layer2 = Prepare_Layer(16, 8)
        self.aggregate_layer3 = Prepare_Layer(8, self.output_dim)

    def forward(self, input, reachable_edge_matrix):
        reachable_edge_matrix = torch.Tensor(reachable_edge_matrix)
        # message passing among nodes
        # the information is flowing from leaves to roots
        # raise x into higher dimension
        output = self.act_fn(self.prepare_layer1(input))
        output = self.act_fn(self.prepare_layer2(output))
        output = self.act_fn(self.prepare_layer3(output))

        for d in range(self.max_depth):
            aggregate_output = self.act_fn(self.aggregate_layer1(output))
            aggregate_output = self.act_fn(self.aggregate_layer2(aggregate_output))
            aggregate_output = self.act_fn(self.aggregate_layer3(aggregate_output))

            aggregate_output = torch.matmul(reachable_edge_matrix[d], aggregate_output)

            # assemble neighboring information
            output = torch.add(aggregate_output, output)
        return output
