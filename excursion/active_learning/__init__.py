import datetime
import torch
import gpytorch
import numpy as np
import math
from scipy.stats import norm
from scipy.linalg import cho_solve
from excursion.utils import h_normal
from torch.distributions import Normal
from excursion.utils import truncated_std_conditional
import time
import os


def MES_gpu(gp, testcase, thresholds, X_grid, device, dtype):

    # compute predictive posterior of Y(x) | train data
    kernel = gp.covar_module
    likelihood = gp.likelihood
    gp.eval()
    likelihood.eval()

    Y_pred_grid = likelihood(gp(X_grid))

    torch.set_printoptions(profile="full")
    #print('############################### Y_pred_grid.variance ', Y_pred_grid.variance)
    print('############################### Y_pred_grid diag covariance_matrix ', torch.diagonal(Y_pred_grid.covariance_matrix))
    

    normal_grid = Normal(
        loc=Y_pred_grid.mean, scale=torch.sqrt(torch.diagonal(Y_pred_grid.covariance_matrix))
    )

    # entropy of S(x_candidate)
    entropy_grid = torch.zeros(X_grid.size()[0],).to(device, dtype)

    for j in range(len(thresholds) - 1):
        # p(S(x)=j)
        p_j = normal_grid.cdf(thresholds[j + 1]) - normal_grid.cdf(thresholds[j]).to(
            device, dtype
        )
        entropy_grid[p_j > 0.0] -= torch.logsumexp(p_j, 0).to(device, dtype) * torch.logsumexp(
            torch.log(p_j), 0
        )

    return entropy_grid


def MES(gp, testcase, thresholds, x_candidate, device, dtype):

    # compute predictive posterior of Y(x) | train data
    kernel = gp.covar_module
    likelihood = gp.likelihood
    gp.eval()
    likelihood.eval()

    Y_pred_candidate = likelihood(gp(x_candidate))

    normal_candidate = torch.distributions.Normal(
        loc=Y_pred_candidate.mean, scale=Y_pred_candidate.variance ** 0.5
    )

    # entropy of S(x_candidate)
    entropy_candidate = torch.Tensor([0.0]).to(device, dtype)

    for j in range(len(thresholds) - 1):
        # p(S(x)=j)
        p_j = normal_candidate.cdf(thresholds[j + 1]) - normal_candidate.cdf(
            thresholds[j]
        )

        if p_j > 0.0:
            # print(x_candidate, p_j,j)

            entropy_candidate -= torch.logsumexp(p_j, 0).to(
                device, dtype
            ) * torch.logsumexp(torch.log(p_j), 0)

    return entropy_candidate.detach()  # .to(device, dtype)


def PES(gp, testcase, thresholds, x_candidate, device, dtype):
    """
    Calculates information gain of choosing x_candidadate as next point to evaluate.
    Performs this calculation with the Predictive Entropy Search approximation. Roughly,
    PES(x_candidate) = int dx { H[Y(x_candidate)] - E_{S(x=j)} H[Y(x_candidate)|S(x=j)] }
    Notation: PES(x_candidate) = int dx H0 - E_Sj H1

    """

    # compute predictive posterior of Y(x) | train data
    kernel = gp.covar_module
    likelihood = gp.likelihood
    gp.eval()
    likelihood.eval()

    X_grid = testcase.X  # .to(device, dtype)
    X_all = torch.cat((x_candidate, X_grid))  # .to(device, dtype)
    Y_pred_all = likelihood(gp(X_all))
    Y_pred_grid = torch.distributions.Normal(
        loc=Y_pred_all.mean[1:], scale=(Y_pred_all.variance[1:]) ** 0.5
    )

    # vector of expected value H1 under S(x) for each x in X_grid
    E_S_H1 = torch.zeros(len(X_grid))  # .to(device, dtype)

    for j in range(len(thresholds) - 1):

        # vector of sigma(Y(x_candidate)|S(x)=j) truncated
        trunc_std_j = truncated_std_conditional(
            Y_pred_all, thresholds[j], thresholds[j + 1]
        )
        H1_j = h_normal(trunc_std_j)

        # vector of p(S(x)=j)
        p_j = Y_pred_grid.cdf(thresholds[j + 1]) - Y_pred_grid.cdf(thresholds[j])
        mask = torch.where(p_j == 0.0)
        H1_j[mask] = 0.0
        E_S_H1 += p_j * H1_j  # expected value of H1 under S(x)

    # entropy of Y(x_candidate)
    H0 = h_normal(Y_pred_all.variance[0] ** 0.5)

    # info grain on the grid, vector
    info_gain = H0 - E_S_H1

    info_gain[~torch.isfinite(info_gain)] = 0.0  # just to avoid NaN

    # cumulative info gain over grid
    cumulative_info_gain = info_gain.sum()

    return cumulative_info_gain.item()


def PPES(gp, testcase, thresholds, x_candidate):
    """
    Calculates information gain of choosing x_candidadate as next point to evaluate.
    Performs this calculation with the Predictive Entropy Search approximation weighted by the posterior. 
    Roughly,
    PES(x_candidate) = int Y(x)dx { H[Y(x_candidate)] - E_{S(x=j)} H[Y(x_candidate)|S(x=j)] }
    Notation: PES(x_candidate) = int dx H0 - E_Sj H1

    """

    # compute predictive posterior of Y(x) | train data
    raise NotImplmentedError(
        "Should be same strcture as PES but the cumulative info gain is weighted"
    )


acquisition_functions = {"PES": PES, "MES": MES, "MES_gpu": MES_gpu}
