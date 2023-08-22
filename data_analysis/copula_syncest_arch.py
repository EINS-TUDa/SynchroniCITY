import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import math
from statsmodels.distributions.copula.archimedean import ClaytonCopula, GumbelCopula, FrankCopula
from statsmodels.distributions.copula.elliptical import StudentTCopula


def plot_cop_density(u, arch_cop_type, log=False):
    plt.figure()
    c = {
        "clayton": ClaytonCopula(),
        "gumbel": GumbelCopula(),
        "frank": FrankCopula(),
    }[arch_cop_type]
    t = np.linspace(.01, 50, 500)
    d = [c.logpdf(u, th) if log else c.pdf(u, th) for th in t]
    if any(np.isnan(d)):
        print("Warning: NaN in density")
    plt.plot(t, d)
    plt.title(f"{arch_cop_type} copula density")
    plt.xlabel("theta")
    plt.ylabel("density")
    if any(np.abs(d) > 1e20) and not log:
        plt.yscale("log")


def most_likely_tau(u, arch_cop_type):
    n = len(u)
    theta_start = {
        "clayton": 2,
        "gumbel": 1.2,
        "frank": 1,
    }[arch_cop_type]
    theta_bounds = {
        "clayton": [1e-3, math.inf],
        "gumbel": [1 + 1e-3, 100],
        "frank": [0.02, 100],
        # "joe": [1, math.inf],
        # "amh": [-1e-3, 1],
    }[arch_cop_type]
    c = {
        "clayton": ClaytonCopula(),
        "gumbel": GumbelCopula(),
        "frank": FrankCopula(),
    }[arch_cop_type]

    def log_likelihood(theta):
        l = c.logpdf(u, theta)
        if np.isinf(l):  # prevent wrong optimization result
            return np.nan
        else:
            return -l

    optimizeResult = minimize(log_likelihood, theta_start, method="powell", bounds=[theta_bounds])
    if not optimizeResult.success:
        print(arch_cop_type, "Warning: optimization failed:", optimizeResult.message, optimizeResult.fun)
        if optimizeResult.fun > -1e3:
            print(optimizeResult)
            print(u)
            return np.nan
    if np.isnan(optimizeResult.fun):
        return np.nan
    if optimizeResult.fun == math.inf:
        print(arch_cop_type, "Warning: optimization result is inf", arch_cop_type, theta_bounds)
        return np.nan
    if not theta_bounds[0] <= optimizeResult.x[0] <= theta_bounds[1]:
        print(optimizeResult)
        print(u)
        print(arch_cop_type, "Warning: optimization result out of bounds", arch_cop_type, theta_bounds)
    return theta2kendtau(arch_cop_type, optimizeResult.x[0])


def theta2kendtau(arch_cop_type, theta):
    if arch_cop_type == 'clayton':
        return ClaytonCopula.tau(None, theta)
    elif arch_cop_type == 'gumbel':
        return GumbelCopula.tau(None, theta)
    elif arch_cop_type == 'frank':
        return FrankCopula.tau(None, theta)
    # elif arch_cop_type == 'joe':
    #     return theta
    # elif arch_cop_type == 'amh':
    #     return theta
    else:
        raise Exception("unknown archimedean copula")
