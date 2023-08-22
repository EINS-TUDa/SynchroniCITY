import numpy as np
from matplotlib import pyplot as plt

from copula_syncest_gaus import most_likly_sync as mls_gaus, most_likely_tau as mlt_gaus
from copula_syncest_arch import most_likely_tau as mlt_arch

us = np.arange(0, 1, .001)


def plot_tau_u1(cop_type, u_others, n):
    estimator = {
        "gaussian": mlt_gaus,
        "gumbel": lambda u: mlt_arch(u, "gumbel"),
        "clayton": lambda u: mlt_arch(u, "clayton"),
    }[cop_type]
    estimate = lambda u1x: estimator([u1x, *[u_others for _ in range(n - 1)]])
    plt.plot(us, list(map(estimate, us)), label=f"{cop_type} n={n} u_oth={u_others}")
    plt.xlabel("u_1")
    plt.ylabel("tau")
    plt.legend()


plot_tau_u1("gaussian", 0.5, 5)
plot_tau_u1("gumbel", 0.5, 5)

plot_tau_u1("gaussian", 0.2, 5)
plot_tau_u1("gaussian", 0.7, 5)

plot_tau_u1("gaussian", 0.5, 25)
plot_tau_u1("gumbel", 0.5, 25)
