import logging

import numpy as np
from torch import nn

logger = logging.getLogger('experiment')


# def mean_nll(logits, y):
#     return nn.functional.binary_cross_entropy_with_logits(logits, y)
#
#
# def mean_accuracy(logits, y):
#     preds = (logits > 0.).float()
#     return ((preds - y).abs() < 1e-2).float().mean()


def mean_nll(logits, y):
    return nn.functional.mse_loss(logits, y)


def mean_accuracy(logits, y):
    preds = (logits > 0).float() - 0.5

    # print(preds)
    return ((preds - y).abs() < 1e-2).float().mean()


# Train loop

def pretty_print(*values):
    col_width = 13

    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    logger.warning("   ".join(str_values))
