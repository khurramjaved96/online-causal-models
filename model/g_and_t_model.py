import logging
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger('experiment')


class RepresentationModel(nn.Module):
    def __init__(self, features, device):
        super(RepresentationModel, self).__init__()
        #
        self.params = []
        self.conv1 = nn.Conv2d(2, 4, 3, 1, 1)

        self.lin1 = nn.Linear(4 * 14 * 14, features)

        self.params.append(self.conv1)
        self.params.append(self.lin1)

        self.total_weights = 0
        self.memory = []
        self.old_vals = {}

        for param in self.params:
            for w in [param.weight, param.bias]:
                logger.info("Feature Weight shape = %s", str(w.shape))
                nn.init.ones_(w)

                if self.total_weights > 1:
                    w.data = torch.bernoulli(torch.zeros_like(w) + 0.25)
                    w.data = w.data + (torch.bernoulli(torch.zeros_like(w) + 0.25) * -1)
                else:
                    w.data = torch.bernoulli(torch.zeros_like(w) + 0.5)
                if len(w.shape) == 1:
                    nn.init.zeros_(w)

                self.total_weights += 1
        for param in self.params:
            for w in [param.weight, param.bias]:
                self.memory.append(torch.ones_like(w))

        self.device = device


    def perturb_feature(self, n, factor):
        type_of = float(random.sample([-1, 0, 1], 1)[0])
        feature_vector = torch.zeros_like(self.lin1.weight.data[n, :])
        feature_vector += factor
        mask = torch.bernoulli(feature_vector)
        self.feature_store = self.lin1.weight.data.clone()
        self.lin1.weight.data[n, :] = (self.lin1.weight.data[n, :]) * (1 - mask) + mask * type_of

    def get_feature(self, n):
        return self.lin1.weight.data[n, :]

    def revert_feature(self, n):
        self.lin1.weight.data[n, :] = self.feature_store[n, :]

    def perturb_layer(self, layer_no, factor):
        type_of = float(random.sample([0, 1], 1)[0])
        assert (factor >= 0 and factor <= 1.0)
        counter = 0
        for param in self.params:
            for w in [param.weight, param.bias]:
                if counter == layer_no:
                    temp_weight = torch.zeros_like(
                        w) + factor

                    temp_weight = (temp_weight > 1).int() + (temp_weight <= 1).int() * temp_weight

                    self.old_vals[counter] = w.data.clone()

                    counter_temp = 0
                    while counter_temp < 20:
                        # Make sure random change does not zero out all the weights. That will result in a degenerate solution to the variance reduction problem -- something we want to avoid.
                        mask = torch.bernoulli(temp_weight)
                        if torch.sum(mask) == 0:
                            total_size = np.prod(mask.shape)
                            index = random.sample(list(range(total_size)), 1)[0]
                            mask.view(-1)[index] = 1

                        if torch.sum(mask) == 0:
                            logger.info("Mask sum is still zero")
                        w.data = (mask * type_of) + w.data * (1 - mask)
                        if torch.sum(self.print_diff()) == 0:
                            pass
                        else:
                            break
                        counter_temp += 1

                counter += 1
                assert (torch.sum(param.bias) == 0)
        if torch.sum(self.params[0].weight) == 0:

            total_size = np.prod(self.params[0].weight.shape)
            while torch.sum(self.params[0].weight) < (total_size / 10):
                index = random.sample(list(range(total_size)), 1)[0]
                self.params[0].weight.data.view(-1)[index] = 1

    def revert_layer(self, layer_no):
        counter = 0
        for param in self.params:
            for w in [param.weight, param.bias]:
                if counter == layer_no:
                    diff = (torch.abs(w.data - self.old_vals[counter]) > 0).float()
                    if torch.sum(diff) > 0:
                        diff = diff / torch.sum(diff)
                    self.memory[counter] += diff
                    w.data = self.old_vals[counter].clone()
                counter += 1

    def print_diff(self):
        return 1 - (self.old_vals[0] == self.params[0].weight).int()

    def binarize(self, x):
        return (x > 0).float()

    def forward(self, input):

        input = self.conv1(input).view(len(input), -1)
        out = F.relu(self.lin1(input.view(input.shape[0], -1)))
        out = self.binarize(out)

        return out


class Linear_Model(nn.Module):
    def __init__(self, width):
        super(Linear_Model, self).__init__()
        lin1 = nn.Linear(width, 1, bias=False)

        for lin in [lin1]:
            nn.init.xavier_uniform_(lin.weight)

        self._main = nn.Sequential(lin1)

    def forward(self, input):
        return self._main(input)
