import logging
import os
import os.path
import random
from collections import namedtuple

logger = logging.getLogger('experiment')
# import cv2
import numpy as np
import torch.nn.functional as F
import utils.g_and_t_utils as pwb

from torch import optim

# import torch.multiprocessing as multi

#
# multi.set_start_method('spawn')

transition = namedtuple('x_traj', 'state, next_state, action, reward, is_terminal')
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_run(model_path):
    import json
    with open(model_path + "/metadata.json") as json_file:
        data = json.load(json_file)
    layers_learn = data["results"]["Layers meta values"]

    return data['params'], layers_learn


def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def resize_image(img, factor):
    '''

    :param img:
    :param factor:
    :return:
    '''
    img2 = np.zeros(np.array(img.shape) * factor)

    for a in range(0, img.shape[0]):
        for b in range(0, img.shape[1]):
            img2[a * factor:(a + 1) * factor, b * factor:(b + 1) * factor] = img[a, b]
    return img2


def get_run(arg_dict, rank=0):
    # print(arg_dict)
    combinations = []

    if isinstance(arg_dict["seed"], list):
        combinations.append(len(arg_dict["seed"]))

    for key in arg_dict.keys():
        if isinstance(arg_dict[key], list) and not key == "seed":
            combinations.append(len(arg_dict[key]))

    total_combinations = np.prod(combinations)
    selected_combinations = []
    for base in combinations:
        selected_combinations.append(rank % base)
        rank = int(rank / base)

    counter = 0
    result_dict = {}

    result_dict["seed"] = arg_dict["seed"]
    if isinstance(arg_dict["seed"], list):
        result_dict["seed"] = arg_dict["seed"][selected_combinations[0]]
        counter += 1
    #

    for key in arg_dict.keys():
        if key != "seed":
            result_dict[key] = arg_dict[key]
            if isinstance(arg_dict[key], list):
                result_dict[key] = arg_dict[key][selected_combinations[counter]]
                counter += 1

    logger.info("Parameters %s", str(result_dict))
    # 0/0
    return result_dict


def log_model(net):
    for name, param in net.named_parameters():
        # print(name)
        if param.meta:
            logger.info("Weight in meta-optimizer = %s %s", name, str(param.shape))
        if param.adaptation:
            logger.debug("Weight for adaptation = %s %s", name, str(param.shape))


#

def collect_data(env, data_points):
    list_of_observations = []
    labels = []
    for step in range(data_points):
        state = env.get_state()
        if step > 0:
            list_of_observations.append(np.array(prev_state))
            labels.append(np.array(state))
        prev_state = state
        env.step()
    return np.array(list_of_observations), np.array(labels)


def take_action(env, data_points):
    list_of_observations = []
    labels = []
    for step in range(data_points):
        state = env.get_state()
        if step > 0:
            list_of_observations.append(np.array(prev_state))
            labels.append(np.array(state))
        prev_state = state
        env.step()
    return np.array(list_of_observations), np.array(labels)


def visualize_data(env, data_points, intervene=False, intervention_frequency=100):
    import cv2
    list_of_observations = []
    labels = []
    for step in range(data_points):
        state = env.get_state()
        if step % intervention_frequency == 0 and intervene:
            env.intervene(1)
            env.intervene(2)
            env.intervene(3)
            env.intervene(4)
            env.intervene(5)
        if step > 0:
            list_of_observations.append(np.array(prev_state))
            labels.append(np.array(state))

        prev_state = state
        env.step()
        img = env.render()
        cv2.imshow("Bouncing_balls", img)
        cv2.waitKey(5)

    return np.array(list_of_observations), np.array(labels)


def make_toy_environment(images, labels, e, device, noise=True, remove=[]):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def torch_xor(a, b):
        return (a - b).abs()  # Assumes both inputs are either 0 or 1

    for a in remove:
        keep = torch.nonzero((labels != a).int())
        images = images[keep].squeeze()
        labels = labels[keep].squeeze()

    y_onehot = torch.FloatTensor(len(labels), 10)
    y_onehot.zero_()

    y_onehot = y_onehot.scatter(1, labels.unsqueeze(1).long(), 1)

    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    #
    labels = (labels >= 5).float()
    if noise:
        labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    colors_one_hot = torch.FloatTensor(len(labels), 2).zero_()

    colors = colors_one_hot.scatter(1, colors.unsqueeze(1).long(), 1)

    features = torch.cat([y_onehot, colors], dim=1)

    # Apply the color to the image by zeroing out the other color channel

    return {
        'images': (features.float()).to(device),
        'labels': labels[:, None].to(device)
    }


def make_environment(images, labels, e, device, noise=True, remove=[]):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def torch_xor(a, b):
        return (a - b).abs()  # Assumes both inputs are either 0 or 1

    for a in remove:
        keep = torch.nonzero((labels != a).int())
        images = images[keep].squeeze()
        labels = labels[keep].squeeze()

    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    if noise:
        labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    labels = labels - 0.5
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor((labels + 0.5), torch_bernoulli(e, len(labels)))
    # print(colors)
    # print(labels)
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
    return {
        'images': (images.float() / 255.).to(device),
        'labels': labels[:, None].to(device)
    }


def sort_data(images, labels):
    dict_images = {}
    for x, y in zip(images, labels):
        y_ = int(y)
        # print(y)

        if y_ in dict_images:
            dict_images[y_].append(x.unsqueeze(0))
        else:
            dict_images[y_] = [x.unsqueeze(0)]

    images_sorted = []
    labels_sorted = []
    for x in dict_images.keys():
        # print(torch.cat(dict_images[x],0).shape)

        images_sorted.append(torch.cat(dict_images[x], 0))
        labels_sorted.append(torch.zeros(len(dict_images[x])) + x)

    images_sorted = torch.cat(images_sorted, 0)
    labels_sorted = torch.cat(labels_sorted, 0)
    counter = 0
    return images_sorted, labels_sorted


def fit_data(rep_model, train_x, train_y, linear_predictor, indices_testing, flags, rate, rank, random_number):
    set_seed(rank * random_number)
    optimizer = optim.Adam(linear_predictor.parameters(), lr=flags['update_lr'])

    weights = (linear_predictor.parameters())[0]
    if rank == 0:
        pass
    else:
        rep_model.perturb_all_layers(rate)

    with torch.no_grad():
        rep_x = rep_model(train_x)

    running_loss = 0
    for a in range(0, 80):
        # print(a)
        indices = random.sample(list(range(len(train_x))), 1024)
        batch_x = rep_x[indices]
        batch_y = train_y[indices]
        for x, y in zip(batch_x, batch_y):
            prediction = linear_predictor(x.unsqueeze(0))
            loss = F.binary_cross_entropy_with_logits(prediction, y.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            running_loss = running_loss * 0.9995 + loss.detach() * 0.0005
            optimizer.step()

    batch_x_test = rep_x[indices_testing]
    batch_y_test = train_y[indices_testing]

    pred_test = linear_predictor(batch_x_test)
    pred_labels = torch.argmax(batch_y_test, dim=1)
    loss = F.binary_cross_entropy_with_logits(pred_test, batch_y_test)
    test_labels = torch.argmax(pred_test, dim=1)

    accuracy = float((pred_labels == test_labels).int().sum()) / float(len(pred_labels))

    return running_loss.detach().item(), linear_predictor, rep_model, accuracy, running_loss.detach()


def fit_data_backup(rep_model, train_x, train_y, linear_predictor, indices_testing, flags, rate, rank, random_number):
    set_seed(rank * random_number)
    optimizer = optim.Adam(linear_predictor.parameters(), lr=flags['update_lr'])

    if rank == 0:
        pass
    else:
        rep_model.perturb_all_layers(rate)

    with torch.no_grad():
        rep_x = rep_model(train_x)

    running_loss = 0
    for a in range(0, 80):
        # print(a)
        indices = random.sample(list(range(len(train_x))), 1024)
        batch_x = rep_x[indices]
        batch_y = train_y[indices]
        for x, y in zip(batch_x, batch_y):
            prediction = linear_predictor(x.unsqueeze(0))
            loss = F.binary_cross_entropy_with_logits(prediction, y.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            running_loss = running_loss * 0.9995 + loss.detach() * 0.0005
            optimizer.step()

    batch_x_test = rep_x[indices_testing]
    batch_y_test = train_y[indices_testing]

    pred_test = linear_predictor(batch_x_test)
    pred_labels = torch.argmax(batch_y_test, dim=1)
    loss = F.binary_cross_entropy_with_logits(pred_test, batch_y_test)
    test_labels = torch.argmax(pred_test, dim=1)

    accuracy = float((pred_labels == test_labels).int().sum()) / float(len(pred_labels))

    return running_loss.detach().item(), linear_predictor, rep_model, accuracy, running_loss.detach()


def fit_data(rep_model, train_x, train_y, linear_predictor, indices_testing, flags, rate, rank, random_number):
    set_seed(rank * random_number)
    optimizer = optim.Adam(linear_predictor.parameters(), lr=flags['update_lr'])

    weights = (linear_predictor.parameters())[0]
    if rank == 0:
        pass
    else:
        rep_model.perturb_all_layers(rate)

    with torch.no_grad():
        rep_x = rep_model(train_x)

    running_loss = 0
    for a in range(0, 80):
        # print(a)
        indices = random.sample(list(range(len(train_x))), 1024)
        batch_x = rep_x[indices]
        batch_y = train_y[indices]
        for x, y in zip(batch_x, batch_y):
            prediction = linear_predictor(x.unsqueeze(0))
            loss = F.binary_cross_entropy_with_logits(prediction, y.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            running_loss = running_loss * 0.9995 + loss.detach() * 0.0005
            optimizer.step()

    batch_x_test = rep_x[indices_testing]
    batch_y_test = train_y[indices_testing]

    pred_test = linear_predictor(batch_x_test)
    pred_labels = torch.argmax(batch_y_test, dim=1)
    loss = F.binary_cross_entropy_with_logits(pred_test, batch_y_test)
    test_labels = torch.argmax(pred_test, dim=1)

    accuracy = float((pred_labels == test_labels).int().sum()) / float(len(pred_labels))

    return running_loss.detach().item(), linear_predictor, rep_model, accuracy, running_loss.detach()


#
#
# def perturb_and_train(rep_model, envs, trainer):
#     rep_model.perturb_layer(0, random.random() / 5)
#
#     x_rep_list = []
#     y_list = []
#
#     for inner_perturbs in range(5):
#
#         with torch.no_grad():
#             for e in envs:
#                 x_rep_list.append(rep_model(e['images']))
#                 y_list.append(e['labels'])
#     set_seed(rank * random_number)
#     optimizer = optim.Adam(linear_predictor.parameters(), lr=flags['update_lr'])
#
#     y_combined = torch.cat(y_list, dim=0)
#     x_rep_combined = torch.cat(x_rep_list, dim=0)
#
#     credit_before_inner = trainer.compute_variance_cheating(flags["l1_penalty"], linear_predictor, x_rep_list, y_list,
#                                                             x_rep_combined, y_combined, device)
#
#
#     if rank == 0:
#         pass
#     else:
#         rep_model.perturb_all_layers(rate)
#
#     with torch.no_grad():
#         rep_x = rep_model(train_x)
#
#     running_loss = 0
#     for a in range(0, 80):
#         # print(a)
#         indices = random.sample(list(range(len(train_x))), 1024)
#         batch_x = rep_x[indices]
#         batch_y = train_y[indices]
#         for x, y in zip(batch_x, batch_y):
#             prediction = linear_predictor(x.unsqueeze(0))
#             loss = F.binary_cross_entropy_with_logits(prediction, y.unsqueeze(0))
#             optimizer.zero_grad()
#             loss.backward()
#             running_loss = running_loss * 0.9995 + loss.detach() * 0.0005
#             optimizer.step()
#
#     batch_x_test = rep_x[indices_testing]
#     batch_y_test = train_y[indices_testing]
#
#     pred_test = linear_predictor(batch_x_test)
#     pred_labels = torch.argmax(batch_y_test, dim=1)
#     loss = F.binary_cross_entropy_with_logits(pred_test, batch_y_test)
#     test_labels = torch.argmax(pred_test, dim=1)
#
#     accuracy = float((pred_labels == test_labels).int().sum()) / float(len(pred_labels))
#
#     return running_loss.detach().item(), linear_predictor, rep_model, accuracy, running_loss.detach()
#

def bound_rate(rate_of_change):
    if rate_of_change > 0.6:
        rate_of_change = 0.6
    if rate_of_change < 0.001:
        rate_of_change = random.random() / 20
    return rate_of_change


def compute_loss(linear_predictor, x_rep_combined, y_combined):
    predictions = linear_predictor(x_rep_combined)
    loss_before = pwb.mean_nll(predictions, y_combined)
    return loss_before


def get_rep(envs, rep_model):
    x_rep_list = []
    y_list = []

    with torch.no_grad():
        for e in envs:
            x_rep_list.append(rep_model(e['images']))
            y_list.append(e['labels'])

    y_combined = torch.cat(y_list, dim=0)
    x_rep_combined = torch.cat(x_rep_list, dim=0)
    return x_rep_combined, y_combined, x_rep_list, y_list


def fit_data_sanity(train_x, train_y, linear_predictor, indices_testing, flags):
    optimizer = optim.Adam(linear_predictor.parameters(), lr=flags['update_lr'])

    rep_x = train_x

    running_loss = 0
    for a in range(0, 80):
        print(a)
        # print(a)
        indices = random.sample(list(range(len(train_x))), 1024)
        batch_x = rep_x[indices]
        batch_y = train_y[indices]
        for x, y in zip(batch_x, batch_y):
            prediction = linear_predictor(x.unsqueeze(0))
            loss = F.binary_cross_entropy_with_logits(prediction, y.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            running_loss = running_loss * 0.9995 + loss.detach() * 0.0005
            optimizer.step()

    batch_x_test = rep_x[indices_testing]
    batch_y_test = train_y[indices_testing]

    pred_test = linear_predictor(batch_x_test)
    pred_labels = torch.argmax(batch_y_test, dim=1)
    loss = F.binary_cross_entropy_with_logits(pred_test, batch_y_test)
    test_labels = torch.argmax(pred_test, dim=1)

    accuracy = float((pred_labels == test_labels).int().sum()) / float(len(pred_labels))

    return running_loss.detach().item(), accuracy

#
#
# def fit_data(rep_model, train_x, train_y, linear_predictor, indices_testing, flags, rate, rank, random_number):
#     # print("Thread ", rank, "started")
#     # return rank*10,2
#
#     set_seed(rank*random_number)
#     optimizer = optim.Adam(linear_predictor.parameters(), lr=flags['update_lr'])
#
#
#
#     if rank == 0:
#         pass
#     else:
#         # for layer in range(0, rep_model.total_weights):
#         rep_model.perturb_all_layers(rate)
#
#     with torch.no_grad():
#         rep_x = rep_model(train_x)
#
#
#     running_loss = 0
#     for a in range(0, 80):
#         # print(a)
#         indices = random.sample(list(range(len(train_x))), 1024)
#         batch_x = rep_x[indices]
#         batch_y = train_y[indices]
#         for x, y in zip(batch_x, batch_y):
#
#             prediction = linear_predictor(x.unsqueeze(0))
#             loss = F.binary_cross_entropy_with_logits(prediction, y.unsqueeze(0))
#             optimizer.zero_grad()
#             loss.backward()
#             running_loss = running_loss*0.9995 + loss.detach()*0.0005
#             optimizer.step()
#
#     # Shift to running loss to see if that also works
#
#     batch_x_test = rep_x[indices_testing]
#     batch_y_test = train_y[indices_testing]
#
#     pred_test = linear_predictor(batch_x_test)
#     pred_labels = torch.argmax(batch_y_test, dim=1)
#     loss = F.binary_cross_entropy_with_logits(pred_test, batch_y_test)
#     test_labels = torch.argmax(pred_test, dim=1)
#
#     accuracy = float((pred_labels == test_labels).int().sum()) / float(len(pred_labels))
#
#
#     return loss.detach().item(), linear_predictor, rep_model, accuracy, running_loss.detach()
#
