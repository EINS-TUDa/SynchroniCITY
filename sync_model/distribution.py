import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from math import *


def plot_dist(d, name="", label=None, cdf=False, log=False, ppfs=0, ppfe=1, start=None, end=None, ax=None,
              metrics=None):
    if not hasattr(d, "pdf"):
        disk = True
        cdf = True
    else:
        disk = False
    if metrics is None:
        metrics = not disk and not log and ax is None
    start, end = start or d.ppf(ppfs), end or d.ppf(ppfe)
    if np.isinf(end):
        end = d.ppf(0.98)
    x = np.linspace(start, end, 5000)
    if not ax:
        fig, ax = plt.subplots(1, 1)
    ax.set_title(name)
    if not disk:
        ax.plot(x, d.pdf(x), label=label or 'pdf')
    if metrics:
        ax.vlines(d.mean(), 0, d.pdf(d.mean()), "k", label="mean")
        ax.vlines(d.median(), 0, d.pdf(d.median()), "grey", label="median")
        ax.hlines(0, d.mean() - d.std(), d.mean() + d.std(), "s", label="std")
    if cdf:
        ax.plot(x, d.cdf(x), label='cdf')
    if log:
        ax.set_yscale('log')
    ax.grid()
    ax.legend()
    return ax


def beta(a=2, b=4, scale=1, plot=False):
    d = stats.beta(a, b, scale=scale)
    if plot:
        plot_dist(d, "beta")
    return d


def beta_EV(E, V, scale=15):
    E = E / scale
    V = V / scale ** 2
    # mean : a/(a+b)*scale
    # var : ab/(a+b)^2/(a+b+1)
    a = E * (E * (1 - E) / V - 1)
    # a=2
    b = a * (1 / E - 1)  # b = a/mean*scale-a = a(scale/mean-1)
    return stats.beta(a, b, scale=scale, loc=0)


def lognormal(o, mu, scale=None, plot=False):
    d = stats.lognorm(s=o, scale=exp(mu) if type(mu) in [float, int] else scale)
    if plot:
        plot_dist(d, "log_normal")
    return d


def lognormal_EV(E, V):
    o2 = log(V / E ** 2 + 1)
    mu = log(E) - o2 / 2
    return lognormal(sqrt(o2), mu)


def e_car(load=11, charging_hrs=1):
    return stats.rv_discrete(values=([0, load], [(24 - charging_hrs) / 24, charging_hrs / 24]))


def plot_dist_2d_ind(p1, p2, ppfs=0, ppfe=1):
    xmin, ymin, xmax, ymax = p1.ppf(ppfs), p2.ppf(ppfs), p1.ppf(ppfe), p2.ppf(ppfe)
    if np.isinf(xmax):
        xmax = p1.ppf(0.98)
    if np.isinf(ymax):
        ymax = p1.ppf(0.98)
    x, y = np.mgrid[0:xmax:.01, 0:ymax:.01]
    pos = np.dstack((x, y))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    z = p1.pdf(x) * p2.pdf(y)
    # ax.contourf(x, y, z )
    cax = ax.imshow(z, extent=(0, xmax, 0, ymax), origin="lower", cmap="viridis")
    plt.colorbar(cax)
