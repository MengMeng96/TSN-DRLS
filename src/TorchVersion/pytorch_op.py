import numpy as np
import torch
from torch.nn.modules import Module
import torch.nn as nn
from torch.nn import functional as F


def glorot(shape, dtype=torch.float32, scope='default'):
    # Xavier Glorot & Yoshua Bengio (AISTATS 2010) initialization (Eqn 16)
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    init = torch.FloatTensor(shape[0], shape[1]).uniform_(-init_range, init_range)
    return torch.Tensor(init)


def leaky_relu(features, alpha=0.2, name=None):
    alpha_feature = torch.mul(alpha, features)
    return torch.maximum(alpha_feature, features)


def ones(shape, dtype=torch.float32, scope='default'):
    init = torch.ones(shape, dtype=dtype)
    return init


def zeros(shape, dtype=torch.float32, scope='default'):
    init = torch.zeros(shape, dtype=dtype)
    return init


class Prepare_Layer(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        super(Prepare_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(glorot([in_features, out_features]))
        self.bias = nn.Parameter(zeros([out_features]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.add(torch.matmul(input, self.weight), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def Criterion(y_pred, adv):
    loss = torch.multiply(torch.log(y_pred + 1e-6), -adv)
    return loss


def discount(rewards, gamma):
    res = [i for i in rewards]
    total = len(rewards)
    for i in reversed(range(len(rewards) - 1)):
        res[i] += gamma * res[-1] * (total - (len(rewards) - i - 1)) / total
    return res


class policy_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, selected_edge_prob, reward, mask):
        # print(selected_edge_prob)
        # print(self.mask)
        # print(selected_edge_prob.shape, self.mask.shape)
        mask = torch.Tensor(mask)
        reward = torch.Tensor([-reward])
        masked_loss = torch.sum(torch.multiply(selected_edge_prob, mask))
        # print(masked_loss.shape, masked_loss)
        logged_loss = torch.log(masked_loss + 1e-6)
        # print("logged_loss", logged_loss)
        reward_loss = torch.multiply(logged_loss, reward)
        # print("######################", reward_loss)
        return reward_loss


def saveFile(fileName, value):
    np.save(fileName, value)


def loadFile(fileName):
    return np.load(fileName, allow_pickle=True).item()
