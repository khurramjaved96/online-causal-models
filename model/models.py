import logging
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger('experiment')

class Linear_Model(nn.Module):
    def __init__(self, width):
        super(Linear_Model, self).__init__()
        lin1 = nn.Linear(width, 1, bias=False)

        for lin in [lin1]:
            nn.init.xavier_uniform_(lin.weight)

        self._main = nn.Sequential(lin1)

    def forward(self, input):
        return self._main(input)


class Linear_Model_MNIST(nn.Module):
    def __init__(self, width):
        super(Linear_Model_MNIST, self).__init__()
        lin1 = nn.Linear(width, 10)

        for lin in [lin1]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        self._main = nn.Sequential(lin1)

    def forward(self, input):
        return self._main(input)


class Linear_Model(nn.Module):
    def __init__(self, width):
        super(Linear_Model, self).__init__()
        lin1 = nn.Linear(width, 1)

        for lin in [lin1]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.ones_(lin.weight)
            nn.init.zeros_(lin.bias)

        self._main = nn.Sequential(lin1)

    def forward(self, input):
        return self._main(input)


class NonLinear_Model_MNIST(nn.Module):
    def __init__(self, width):
        super(NonLinear_Model_MNIST, self).__init__()
        self.lin1 = nn.Linear(1 * 28 * 28, width)
        self.lin2 = nn.Linear(width, 10)

        for lin in [self.lin1, self.lin2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, input):
        input = input.view(input.shape[0], -1)
        input = F.relu(self.lin1(input))
        input = self.lin2(input)

        return input
