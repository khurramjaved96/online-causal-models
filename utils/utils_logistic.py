import logging

import numpy as np
import torch
from torch import nn, autograd

logger = logging.getLogger('experiment')


def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()


def penalty(logits, y):
    scale = torch.tensor(1.).requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def pretty_print(*values):
    col_width = 13

    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    logger.warning("   ".join(str_values))




class Linear_Model(nn.Module):
    def __init__(self, width):
        super(Linear_Model, self).__init__()
        lin1 = nn.Linear(width, 1)

        for lin in [lin1]:
            nn.init.xavier_uniform_(lin.weight)
            # nn.init.xeros_(lin.weight)
            nn.init.zeros_(lin.bias)

        self._main = nn.Sequential(lin1)

    def forward(self, input):
        return self._main(input)

