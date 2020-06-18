import copy
import logging

import numpy as np
import torch
from torch import optim, autograd
from torchvision import datasets

import configs.our_method_exp_1 as mnist
import utils
from environment import toy
from experiment.experiment import experiment
from model import models
from utils import utils_logistic

p = mnist.Parser()
total_seeds = len(p.parse_known_args()[0].seed)
rank = p.parse_known_args()[0].rank
all_args = vars(p.parse_known_args()[0])
print("All args = ", all_args)

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

final_train_accs = []
final_test_accs = []

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

masks = torch.zeros(1, 12).to(device)
list_temp = [int(x) for x in flags["less_likely"].split(",")]
game = toy.Toy(list_temp)

linear_predictor = models.Linear_Model(12).to(device)

mean = torch.zeros_like(list(linear_predictor.parameters())[0])
grad_mean = torch.zeros_like(list(linear_predictor.parameters())[0])
max_weight = torch.zeros_like(list(linear_predictor.parameters())[0])
grad_mean_quick = torch.zeros_like(list(linear_predictor.parameters())[0])
diff_mean_quick = torch.zeros_like(list(linear_predictor.parameters())[0])
grad_total = torch.zeros_like(list(linear_predictor.parameters())[0])
variance = torch.zeros_like(list(linear_predictor.parameters())[0])

x_data = []
y_data = []

optimizer = optim.Adam(list(linear_predictor.parameters()), lr=flags["update_lr"])

for step in range(flags['steps']):

    game.take_action(1)
    x = game.get_state().float().to(device)
    y = game.get_target().float().to(device)

    #
    prediction = linear_predictor(x * torch.sigmoid(masks))
    mean_old = copy.deepcopy(mean)
    mean = 0.99997 * mean + ((0.00003 * list(linear_predictor.parameters())[0].detach()) * x) + ((
                                                                                                         1 - x) * 0.00003 * mean)
    variance = 0.9999 * variance + (0.0001 * (
            (mean - list(linear_predictor.parameters())[0].detach()) * (
            mean_old - list(linear_predictor.parameters())[0].detach())) * x) + (0.0001 * variance * (1 - x))

    train_nll = utils_logistic.mean_nll(prediction, y)
    grad_temp = autograd.grad(train_nll, list(linear_predictor.parameters())[0], create_graph=True,
                              retain_graph=True)[0]

    grad_mean = 0.9999 * grad_mean + 0.0001 * ((grad_temp.detach()) ** 2) * x + (
            1 - x) * 0.0001 * grad_mean

    grad_mean_quick = 0.999 * grad_mean_quick + 0.001 * (grad_temp.detach()) * x + (1 - x) * grad_mean_quick * 0.001

    grad_total += (torch.abs(variance.detach()))

    train_nll_org = train_nll.detach().clone()
    l1 = torch.tensor(0.).to(device)
    for w in linear_predictor.parameters():
        l1 += torch.abs(w).sum()
    train_nll = train_nll + flags["l1_penalty"] * l1

    pre_weights = list(linear_predictor.parameters())[0].detach().clone()
    optimizer.zero_grad()
    train_nll.backward()
    optimizer.step()
    post_weights = list(linear_predictor.parameters())[0].detach().clone()
    # print(post_weights)
    dif_temp = (post_weights - pre_weights) / flags["update_lr"]

    diff_mean_quick = 0.999 * diff_mean_quick + 0.001 * (dif_temp.detach()) * x + (
            1 - x) * diff_mean_quick * 0.001

    with torch.no_grad():
        if step >= 5000000:
            normed = variance.detach() / variance.detach().norm()
            masks += (torch.mean(normed) + flags['alpha'] * (torch.mean(normed)) - normed)

        if step % 10000 == 0:
            logger.info("Mean = %s %s", str(mean.squeeze()), "\t")
            logger.info("L1 loss = %s", str(l1.item() * flags["l1_penalty"]))
            logger.info("NLL loss = %s", str(train_nll_org.item()))
            logger.info("Total loss = %s", str(train_nll.item()))
            logger.info("Variance = %s %s", str(variance.squeeze()), "\t")
            logger.info("Grad mean = %s %s", str(grad_mean.squeeze()), "\t")
            logger.info("Grad mean quick = %s %s", str(grad_mean_quick.squeeze()), "\t")
            logger.info("Diff mean quick = %s %s", str(diff_mean_quick.squeeze()), "\t")
            logger.info("Grad total = %s %s", str(grad_total.squeeze()), "\t")
            logger.info("Weight = %s %s", str(list(linear_predictor.parameters())[0]), "\t")
            logger.debug("Mask = %s %s", str(torch.sigmoid(masks).squeeze())
                         , "\t")
            logger.info("Alpha %s", str(flags['alpha']))

            with torch.no_grad():
                other_logits = test_set['images']
                other_prediction = linear_predictor(other_logits * torch.sigmoid(masks))
                other_accuracy = utils_logistic.mean_accuracy(other_prediction, test_set['labels'])

                other_logits_1 = train_1['images']
                other_prediction = linear_predictor(other_logits_1 * torch.sigmoid(masks))
                other_accuracy_1 = utils_logistic.mean_accuracy(other_prediction, train_1['labels'])

                other_logits_2 = train_2['images']
                other_prediction = linear_predictor(other_logits_2 * torch.sigmoid(masks))
                other_accuracy_2 = utils_logistic.mean_accuracy(other_prediction, train_2['labels'])

    if step % 10000 == 0:
        utils_logistic.pretty_print(
            np.int32(step),
            other_accuracy.detach().cpu().numpy(),
            other_accuracy_1.detach().cpu().numpy(),
            other_accuracy_2.detach().cpu().numpy(),
        )
