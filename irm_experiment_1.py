#Neurips code submission

import logging

import numpy as np
import torch
from torch import optim
from torchvision import datasets

import configs.irm_exp_1 as mnist
import utils
from environment import toy
from experiment.experiment import experiment
from model import models
from utils import utils_logistic

p = mnist.Parser()
total_seeds = len(p.parse_known_args()[0].seed)
rank = p.parse_known_args()[0].rank
all_args = vars(p.parse_known_args()[0])

flags = utils.get_run(vars(p.parse_known_args()[0]), rank)

utils.set_seed(flags["seed"])

my_experiment = experiment(flags["name"], flags, flags['output_dir'], commit_changes=False,
                           rank=int(rank / total_seeds),
                           seed=total_seeds)

my_experiment.results["all_args"] = all_args

logger = logging.getLogger('experiment')

logger.info("Selected args %s", str(flags))

toy_env = toy.Toy()

device = torch.device('cpu')

mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_val = (mnist.data[50000:], mnist.targets[50000:])

rng_state = np.random.get_state()
np.random.shuffle(mnist_train[0].numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist_train[1].numpy())

train_1 = utils.make_toy_environment(mnist_train[0][::2], mnist_train[1][::2], 0.1, device,
                                     noise=not flags["no_noise"])
train_2 = utils.make_toy_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.2, device,
                                     noise=not flags["no_noise"])
test_set = utils.make_toy_environment(mnist_val[0], mnist_val[1], 0.9, device, noise=not flags["no_noise"])

utils_logistic.pretty_print('step', 't1 t1', "t1 t2", 't2 t2', "t2 t1", 'test acc')

masks = torch.zeros(1, 12).to(device).requires_grad_()
list_temp = [int(x) for x in flags["less_likely"].split(",")]

game = toy.Toy(list_temp)
linear_predictor = models.Linear_Model(12).to(device)

optimizer = optim.Adam([masks], lr=flags["update_lr"])

for step in range(flags['steps']):
    losses = []
    for env in range(2):

        game.force_change()
        x_list = []
        y_list = []
        for _ in range(1024):
            game.set_random_state()
            x_list.append(game.get_state().float().to(device))
            y_list.append(game.get_target().float().to(device))

        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        x_mask = x.mean(dim=0).unsqueeze(0)

        prediction = linear_predictor(x * masks)

        train_nll = utils_logistic.mean_nll(prediction, y)
        grad_temp = utils_logistic.penalty(prediction, y)

        train_nll_org = train_nll.detach().clone()
        l1 = torch.tensor(0.).to(device)
        for w in linear_predictor.parameters():
            l1 += torch.abs(w).sum()

        if step > 2000:
            train_nll = train_nll + flags["l1_penalty"] * l1 + grad_temp * flags["grad_penalty"]
        else:
            train_nll = train_nll + flags["l1_penalty"] * l1

        losses.append(train_nll)

    loss_final = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss_final.backward()
    optimizer.step()

    with torch.no_grad():

        if step % 10 == 0:
            with torch.no_grad():
                other_prediction = linear_predictor(test_set['images'] * masks)
                other_accuracy = utils_logistic.mean_accuracy(other_prediction, test_set['labels'])

                other_prediction = linear_predictor(train_1['images'] * masks)
                other_accuracy_1 = utils_logistic.mean_accuracy(other_prediction, train_1['labels'])

                other_prediction = linear_predictor(train_2['images'] * masks)
                other_accuracy_2 = utils_logistic.mean_accuracy(other_prediction, train_2['labels'])

            utils_logistic.pretty_print(
                np.int32(step),
                other_accuracy.detach().numpy(),
                other_accuracy_1.detach().numpy(),
                other_accuracy_2.detach().numpy(),
            )
