import copy
import random

import torch
from torchvision import datasets

import configs.pwb as params
import model.g_and_t_model as model
import trainer.g_and_t_trainer as trainer
import utils
from experiment.experiment import experiment
from utils.g_and_t_utils import *

p = params.PwbParameters()
total_seeds = len(p.parse_known_args()[0].seed)
rank = p.parse_known_args()[0].rank
all_args = vars(p.parse_known_args()[0])
print("All hyperparameters = ", all_args)

flags = utils.get_run(vars(p.parse_known_args()[0]), rank)

utils.set_seed(flags["seed"])

my_experiment = experiment(flags["name"], flags, flags['output_dir'], commit_changes=False,
                           rank=int(rank / total_seeds),
                           seed=total_seeds)

my_experiment.results["all_args"] = all_args

logger = logging.getLogger('experiment')

logger.info("Selected hyperparameters %s", str(flags))

gpu_to_use = rank % flags["gpus"]
if torch.cuda.is_available() and not flags["no_gpu"]:
    device = torch.device('cuda:' + str(gpu_to_use))
    logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
else:
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

env_2_val = flags['env1']
env_1_val = flags['env2']
env_test_val = 0.9

envs = [
    utils.make_environment(mnist_train[0][::2], mnist_train[1][::2], env_1_val, device),
    utils.make_environment(mnist_train[0][1::2], mnist_train[1][1::2], env_2_val, device)
]

test_envs = [
    utils.make_environment(mnist_val[0], mnist_val[1], env_test_val, device)
]

logger.info("Seen MDP latent value %f", env_1_val)
logger.info("Seen MDP latent value %f", env_2_val)
logger.info("Unseen MDP latent value %f", env_test_val)

rep_model = model.RepresentationModel(flags["features"], device)
linear_predictor = model.Linear_Model(flags["features"]).to(device)

rate_of_change = 0.1
train_loss_list = []
train_accuracy_list = []
test_accuracy_list = []
test_loss_list = []
credit_before = None
accepted = 0
for rep_steps in range(flags['steps']):

    x_rep_combined, y_combined, x_rep_list, y_list = utils.get_rep(envs, rep_model)

    credit_before, absolute_weights = trainer.compute_variance_cheating(flags["l1_penalty"], linear_predictor,
                                                                        x_rep_list, y_list,
                                                                        x_rep_combined, y_combined, device)

    magnitude = list(linear_predictor.parameters())[0]
    max_v_arg = torch.argmax(credit_before.squeeze())
    min_mag_arg = torch.argmin(magnitude.squeeze())

    max_v_val_before = credit_before.squeeze()[max_v_arg]
    min_mag_val_before = magnitude.squeeze()[min_mag_arg]
    credit_before = torch.sum(torch.abs(credit_before) ** 2)

    loss_before = utils.compute_loss(linear_predictor, x_rep_combined, y_combined)

    if rep_steps % 10 == 0:
        rep_model.perturb_layer(0, 0.2 * random.random())
        credit_before_real = copy.deepcopy(credit_before)
    else:
        rep_model.perturb_feature(min_mag_arg, 0.2 * random.random())

    x_rep_combined, y_combined, x_rep_list, y_list = utils.get_rep(envs, rep_model)

    # Cheating to speed up the experiment; we could compute this online, but that would take longer.
    credit_after, weights_after = trainer.compute_variance_cheating(flags["l1_penalty"], linear_predictor, x_rep_list,
                                                                    y_list,
                                                                    x_rep_combined, y_combined, device)

    max_v_arg_after = torch.argmax(credit_after.squeeze())

    min_mag_arg_after = torch.argmin(weights_after.squeeze())

    max_v_val_after = credit_after.squeeze()[max_v_arg]
    min_mag_val_after = weights_after.squeeze()[min_mag_arg]

    credit_after = torch.sum(torch.abs(credit_after) ** 2)

    if rep_steps % 10 == 0:
        if credit_after > credit_before:

            rep_model.revert_layer(0)
        else:
            accepted += 1

    else:
        if min_mag_val_before > min_mag_val_after:
            rep_model.revert_feature(min_mag_arg)
        else:
            accepted += 0

    if rep_steps % 50 == 0:
        logger.info("\n")
        logger.debug("### Step = %d ###", rep_steps)
        with torch.no_grad():
            train_acc = []
            test_acc = []
            train_loss = []

            for e in envs:
                other_logits = rep_model(e['images'])
                other_prediction = linear_predictor(other_logits)
                train_acc.append(mean_accuracy(other_prediction, e['labels']))
                train_loss.append(mean_nll(other_prediction, e['labels']))

            other_logits = rep_model(test_envs[0]['images'])
            other_prediction = linear_predictor(other_logits)
            test_acc.append(mean_accuracy(other_prediction, test_envs[0]['labels']))

        logger.info("Accuracy Seen MDP: %s, Accuracy Unseen MDP: %s", str(torch.tensor(train_acc)), str(test_acc))

        train_accuracy_list.append(torch.mean(torch.tensor(train_acc)).item())
        test_accuracy_list.append(test_acc[0].item())
        train_loss_list.append(torch.mean(torch.tensor(train_loss)).item())

    if rep_steps % 1000 == 0:
        my_experiment.results["train_loss"] = train_loss_list
        my_experiment.results["test_accuracy"] = test_accuracy_list
        my_experiment.results["train_accuracy"] = train_accuracy_list
        my_experiment.store_json()
        torch.save(rep_model, my_experiment.path + "rep_net.model")
#
