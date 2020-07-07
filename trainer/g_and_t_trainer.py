from utils.g_and_t_utils import *
from sklearn import linear_model
import random
import utils
import torch.nn.functional as F
import copy
import torch
from torch import optim
def compute_variance(x_data, y_data, model, inner_lr, l1_penalty, steps=50000):
    optimizer = optim.Adam([list(model.parameters())[0]], lr=inner_lr)

    # mean = torch.zeros_like(list(model.parameters())[0])
    # variance = torch.zeros_like(list(model.parameters())[0])
    # variance_total = torch.zeros_like(list(model.parameters())[0])
    # mean.data = list(model.parameters())[0].detach()
    #
    # env = 0
    # counter = 0
    # for env in range(2):
    #     x = x_data[env]
    #     y = y_data[env]
    #
    #     for inner in range(0, 500):
    #         if inner_steps % 1000 == 0:
    #             env = (env+1)%2
    #
    #
    #
    #     x = x_data[env][inner_steps].unsqueeze(0)
    #     y = y_data[env][inner_steps].unsqueeze(0)
    #     prediction = model(x)
    #
    #     mean = 0.99999 * mean + ((0.00001 * list(model.parameters())[0].detach()) * x) + ((
    #                                                                                               1 - x) * 0.00001 * mean)
    #
    #     variance = 0.9999 * variance + (0.0001 * (
    #             (mean - list(model.parameters())[0].detach()) ** 2) * x) + (0.0001 * variance * (1 - x))
    #
    #     train_nll = mean_nll(prediction, y)
    #     grad_temp = autograd.grad(train_nll, list(model.parameters())[0], create_graph=True,
    #                               retain_graph=True)[0]
    #
    #     variance_total += (torch.abs(variance.detach()))
    #
    #
    #     train_nll_org = train_nll.detach().clone()
    #     l1 = None
    #     for w in model.parameters():
    #         if l1 is None:
    #             l1 = torch.abs(w).sum()
    #         else:
    #             l1 += torch.abs(w).sum()
    #     train_nll = train_nll + l1_penalty * l1
    #
    #
    #     optimizer.zero_grad()
    #     train_nll.backward()
    #     optimizer.step()
    #
    #     #
    #     # if inner_steps % 10000 == 0:
    #     #     logger.info("Mean = %s %s", str(mean.squeeze()), "\t")
    #     #     logger.info("NLL loss = %s", str(train_nll_org.item()))
    #     #     logger.info("Total loss = %s", str(train_nll.item()))
    #     #     logger.info("Variance = %s %s", str(variance.squeeze()), "\t")
    #     #     logger.info("Total variance = %s %s", str(variance_total.squeeze()), "\t")
    #
    # return variance_total

def compute_variance_cheating(l1_penalty, linear_predictor,  x_rep_l, y_l, x, y, device):
    #
    clf = linear_model.Ridge(alpha=l1_penalty * len(y), tol=0.0001,
                             max_iter=10000, fit_intercept=False)

    clf.fit(x_rep_l[0].cpu().numpy(), y_l[0].squeeze().cpu().numpy())
    weight_1 = torch.tensor(clf.coef_).float().to(device)

    clf.fit(x_rep_l[1].cpu().numpy(), y_l[1].squeeze().cpu().numpy())
    weight_2 = torch.tensor(clf.coef_).float().to(device)

    variance_total = (torch.abs(weight_1 - weight_2))

    clf.fit(x.cpu().numpy(), y.squeeze().cpu().numpy())

    list(linear_predictor.parameters())[0].data = torch.tensor(clf.coef_).unsqueeze(0).float().to(device)

    absolute_weights = torch.abs(list(linear_predictor.parameters())[0]).detach()

    return variance_total.squeeze(), absolute_weights



def compute_variance_online(iteration, envs, rep_model, l1_penalty, linear_predictor, flags, rank, random_number, type_of):

    # x_all, y_all, x_rep_l, y_l = utils.get_rep(envs, rep_model)

    variance = torch.zeros_like(list(linear_predictor.parameters())[0])
    mean = list(linear_predictor.parameters())[0].data.clone()

    utils.set_seed((rank+1)*random_number)

    optimizer = optim.Adam(linear_predictor.parameters(), lr=flags['update_lr'])
    #
    running_loss = 0

    magnitude = list(linear_predictor.parameters())[0]
    min_mag_arg = torch.argmin(magnitude.squeeze())

    if rank == 0:
        pass
    else:
        if iteration % 2 == 0:
            p = random.random()
            if rank == 2 or rank == 3:
                p = 1
            rep_model.perturb_layer(0, 1.0 * p, rank%2)
        else:

            rep_model.perturb_feature(min_mag_arg, 0.2 * random.random())

    x_all, y_all, x_rep_l, y_l = utils.get_rep(envs, rep_model)

    clf = linear_model.Ridge(alpha=l1_penalty * len(y_l[0]), fit_intercept=False, tol=0.0001,
                             max_iter=10000)
    #
    clf.fit(x_rep_l[0].cpu().numpy(), y_l[0].squeeze().cpu().numpy())

    weight_1 = torch.tensor(clf.coef_).float()

    clf.fit(x_rep_l[1].cpu().numpy(), y_l[1].squeeze().cpu().numpy())

    weight_2 = torch.tensor(clf.coef_).float()

    variance_comparison = torch.abs(weight_1 - weight_2) ** 2
    weights_normal = []
    for counter in range(2):
        for step in range(100):
            if step%10 == 0 and rank ==0:
                print("Step first part:", step,  counter)
            sample_indices = np.sort(random.sample(list(range(len(x_rep_l[counter]))), 10000))

            # if rank == 0:
            #     print("Counter = ", counter)
            #     print("Images shape = ", (x_rep_l[counter].shape))
            #
            #
            # print(sample_indices)
            # quit()
            batch_x = x_rep_l[counter][sample_indices]
            batch_y = y_l[counter][sample_indices]
            #
            for x, y in zip(batch_x, batch_y):

                prediction = linear_predictor(x.unsqueeze(0))
                loss = F.mse_loss(prediction, y.unsqueeze(0))
                l1 = None
                for w in linear_predictor.parameters():
                    l1 = (w.squeeze() ** 2).sum()

                # print(loss,l1_penalty, l1)
                loss = loss + l1_penalty * l1

                # print(x)
                # quit()

                optimizer.zero_grad()
                loss.backward()
                running_loss = running_loss * 0.9995 + loss.detach() * 0.0005
                optimizer.step()

                mean_old = copy.deepcopy(mean)
                mean = 0.99999 * mean + ((0.00001 * list(linear_predictor.parameters())[0].detach()) * x) + ((
                                                                                                                     1 - x) * 0.00001 * mean)

                variance = 0.9999 * variance + (0.0001 * (
                        (mean - list(linear_predictor.parameters())[0].detach()) * (
                        mean_old - list(linear_predictor.parameters())[0].detach())) * x) + (
                                   0.0001 * variance * (1 - x))

        weights_normal.append(list(linear_predictor.parameters())[0].data.clone())
    weight_div_proper = torch.abs(weights_normal[0] - weights_normal[1])**2

    mean = list(linear_predictor.parameters())[0].data.clone()

    for step in range(100):
        for counter in range(len(x_rep_l)):

            if step%10 == 0 and rank ==0:
                print("Step second part: ", step)

            sample_indices = np.sort(random.sample(list(range(len(x_rep_l[counter]))), 10000))

            # if rank == 0:
            #     print("Counter = ", counter)
            #     print("Images shape = ", (x_rep_l[counter].shape))
        #
        #
        # print(sample_indices)
        # quit()
            batch_x = x_rep_l[counter][sample_indices]
            batch_y = y_l[counter][sample_indices]
            #
            for x, y in zip(batch_x, batch_y):

                prediction = linear_predictor(x.unsqueeze(0))
                loss = F.mse_loss(prediction, y.unsqueeze(0))
                l1 = None
                for w in linear_predictor.parameters():
                    l1 = (w.squeeze()**2).sum()


                # print(loss,l1_penalty, l1)
                loss = loss + l1_penalty * l1

                # print(x)
                # quit()

                optimizer.zero_grad()
                loss.backward()
                running_loss = running_loss*0.9995 + loss.detach()*0.0005
                optimizer.step()

                mean_old = copy.deepcopy(mean)
                mean = 0.99999 * mean + ((0.00001 * list(linear_predictor.parameters())[0].detach()) * x) + ((
                                                                                  1 - x) * 0.00001 * mean)

                variance = 0.9999 * variance + (0.0001 * (
                        (mean - list(linear_predictor.parameters())[0].detach()) * (
                            mean_old - list(linear_predictor.parameters())[0].detach())) * x) + (
                                       0.0001 * variance * (1 - x))



    # print("Rank = ", rank, "Weights =", torch.abs(list(linear_predictor.parameters())[0]).flatten().detach())
    # print("Variance = ", variance)
    absolute_weights = torch.abs(list(linear_predictor.parameters())[0]).flatten().detach()[min_mag_arg]
    #
    # print("CLF weight", weight_2)
    # print("learned weight",list(linear_predictor.parameters())[0])
    # print(list(linear_predictor.parameters())[0])
    # print(list(linear_predictor.parameters())[1])

    return torch.sum(variance).detach(), absolute_weights.detach(), copy.deepcopy(rep_model), copy.deepcopy(linear_predictor), torch.sum(weight_div_proper).detach()




def compute_variance_online_cheat(iteration, envs, rep_model, l1_penalty, linear_predictor, flags, rank, random_number, type_of):

    # x_all, y_all, x_rep_l, y_l = utils.get_rep(envs, rep_model)


    variance = torch.zeros_like(list(linear_predictor.parameters())[0])
    mean = list(linear_predictor.parameters())[0].data.clone()

    utils.set_seed((rank+1)*random_number)
    #
    optimizer = optim.Adam(linear_predictor.parameters(), lr=flags['update_lr'])
    #
    running_loss = 0

    magnitude = list(linear_predictor.parameters())[0]
    min_mag_arg = torch.argmin(magnitude.squeeze())

    if rank == 0:
        # print(magnitude)
        pass
    else:
        if iteration % 2 == 0:
            p = random.random()
            if rank == 2:
                p = 1
            rep_model.perturb_layer(0, 1.0 * p, type_of)

        else:
            # print("Feature perturbed")
            # print("Rank, ranodm", rank, 0.2 * random.random())
            rep_model.perturb_feature(min_mag_arg, 0.2 * random.random())

    # print(torch.sum(rep_model.get_feature(min_mag_arg)))
    x_all, y_all, x_rep_l, y_l = utils.get_rep(envs, rep_model)

    clf = linear_model.Ridge(alpha=l1_penalty * len(y_all), fit_intercept= False, tol=0.0001,
                             max_iter=10000)
    #
    clf.fit(x_rep_l[0].cpu().numpy(), y_l[0].squeeze().cpu().numpy())

    weight_1 = torch.tensor(clf.coef_).float()
    #
    clf.fit(x_rep_l[1].cpu().numpy(), y_l[1].squeeze().cpu().numpy())

    weight_2 = torch.tensor(clf.coef_).float()

    variance = torch.abs(weight_1 - weight_2)**2

    clf.fit(x_all.cpu().numpy(), y_all.squeeze().cpu().numpy())
    # print(torch.tensor(clf.coef_).unsqueeze(0).float())
    list(linear_predictor.parameters())[0].data = torch.tensor(clf.coef_).unsqueeze(0).float()
    list(linear_predictor.parameters())[1].data = torch.tensor([0]).float()


    absolute_weights = torch.abs(list(linear_predictor.parameters())[0]).squeeze().detach()[min_mag_arg]


    return torch.sum(variance).detach(), absolute_weights.detach(), copy.deepcopy(rep_model), copy.deepcopy(linear_predictor)