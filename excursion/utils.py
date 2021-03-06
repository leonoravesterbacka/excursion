from scipy.stats import norm
import numpy as np
import torch
import importlib

def cdf(value):
    return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))


def load_example(example):
    testcase = None
    if example == "1Dtoyanalysis":
        testcase = importlib.import_module("excursion.testcases.fast_1D")
    elif example == "1D_test":
        testcase = importlib.import_module("excursion.testcases.1D_test")
    elif example == "2D_test":
        testcase = importlib.import_module("excursion.testcases.2D_test")
    elif example == "3D_test":
        testcase = importlib.import_module("excursion.testcases.3D_test")
    elif example == "2Dtoyanalysis":
        testcase = importlib.import_module("excursion.testcases.fast_2D")
    elif example == "darkhiggs":
        testcase = importlib.import_module("excursion.testcases.darkhiggs")
    elif example == "checkmate":
        testcase = importlib.import_module("excursion.testcases.checkmate")
    elif example == "3dfoursheets":
        testcase = importlib.import_module("excursion.testcases.toy3d_foursheets")
    elif example == "3Dtoyanalysis":
        testcase = importlib.import_module("excursion.testcases.fast_3D")
    elif example.startswith("parabola_"):
        n = [int(s) for s in example if s.isdigit()][0]
        #make_parabola_script(n)
        testcase = importlib.import_module("excursion.testcases.parabola_"+str(n)+"D")
    #elif example == "parabola_1D":
    #    testcase = importlib.import_module("excursion.testcases.parabola_1D")
    #elif example == "parabola_2D":
    #    testcase = importlib.import_module("excursion.testcases.parabola_2D")
    #elif example == "parabola_3D":
    #    testcase = importlib.import_module("excursion.testcases.parabola_3D")
    #elif example == "parabola_4D":
    #    testcase = importlib.import_module("excursion.testcases.parabola_4D")
    else:
        raise RuntimeError("unnkown test case")
    return testcase


def point_entropy(mu_stds, thresholds):
    thresholds = np.concatenate([[-np.inf], thresholds, [np.inf]])

    entropies = []
    for mu, std in mu_stds:
        entropy = 0
        for j in range(len(thresholds) - 1):
            p_within = norm(mu, std).cdf(thresholds[j + 1]) - norm(mu, std).cdf(
                thresholds[j]
            )
            p_within[p_within < 1e-9] = 1e-9
            p_within[p_within > 1 - 1e-9] = 1 - 1e-9
            entropy -= p_within * np.log(p_within)
        entropies.append(entropy)
    return np.mean(np.stack(entropies), axis=0)


def point_entropy_gpytorch(mu_stds, thresholds):
    thresholds = np.concatenate([[-np.inf], thresholds, [np.inf]])

    entropies = []
    for obs_pred in mu_stds:
        entropy = 0
        for j in range(len(thresholds) - 1):
            p_within = norm(
                obs_pred.mean.detach().numpy(), obs_pred.stddev.detach().numpy()
            ).cdf(thresholds[j + 1]) - norm(
                obs_pred.mean.detach().numpy(), obs_pred.stddev.detach().numpy()
            ).cdf(
                thresholds[j]
            )
            p_within[p_within < 1e-9] = 1e-9
            p_within[p_within > 1 - 1e-9] = 1 - 1e-9
            entropy -= p_within * np.log(p_within)
        entropies.append(entropy)
    return np.mean(np.stack(entropies), axis=0)


def mesh2points(grid, npoints_tuple):
    ndim = len(npoints_tuple)
    X = np.moveaxis(grid, 0, ndim).reshape(int(np.product(npoints_tuple)), ndim)
    return X


def points2mesh(X, npoints_tuple):
    ndim = len(npoints_tuple)
    grid = np.moveaxis(X.reshape(*(npoints_tuple + [ndim,])), ndim, 0)
    return grid


def mgrid(rangedef):
    _rangedef = np.array(rangedef, dtype="complex128")
    slices = [slice(*_r) for _r in _rangedef]
    return np.mgrid[slices]


def values2mesh(values, rangedef, invalid, invalid_value=np.nan):
    grid = mgrid(rangedef)
    allX = mesh2points(grid, rangedef[:, 2])
    allv = np.zeros(len(allX))
    inv = invalid(allX)

    if torch.cuda.is_available() and type(values) == torch.Tensor:
        allv[~inv] = values.cpu()
    else:
        allv[~inv] = values

    if np.any(inv):
        allv[inv] = invalid_value
    return allv.reshape(*map(int, rangedef[:, 2]))


def h_normal(var):
    return torch.log(var * (2 * np.e * np.pi) ** 0.5)


def normal_pdf(x):
    return 1.0 / (2 * np.pi) ** 0.5 * torch.exp(-0.2 * x ** 2)


def truncated_std_conditional(Y_pred_all, a, b):
    mu_grid = Y_pred_all.mean[1:]
    std_grid = Y_pred_all.variance[1:] ** 0.5
    mu_candidate = Y_pred_all.mean[0]
    std_candidate = Y_pred_all.variance[0] ** 0.5
    rho = Y_pred_all.covariance_matrix[0, 1:] / (std_candidate * std_grid)

    # norm needs to be a normal distribution but in python
    normal = torch.distributions.Normal(loc=0, scale=1)
    alpha = (a - mu_grid) / std_grid
    beta = (b - mu_grid) / std_grid
    c = normal.cdf(beta) - normal.cdf(alpha)

    # phi(beta) = normal(0,1) at x = beta
    beta_phi_beta = beta * normal_pdf(beta)
    beta_phi_beta[~torch.isfinite(beta_phi_beta)] = 0.0
    alpha_phi_alpha = alpha * normal_pdf(alpha)
    alpha_phi_alpha[~torch.isfinite(alpha_phi_alpha)] = 0.0

    # unnormalized
    first_moment = mu_candidate - std_candidate * rho / c * (
        normal_pdf(beta) - normal_pdf(alpha)
    )

    second_moment = (
        std_candidate ** 2 * (1 - rho ** 2 / c) * (beta_phi_beta - alpha_phi_alpha)
        - mu_candidate ** 2
        + 2 * mu_candidate * first_moment
    )

    return second_moment ** 0.5
