import torch
from sklearn import linear_model


def compute_variance_cheating(l1_penalty, linear_predictor, x_rep_l, y_l, x, y, device):
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
