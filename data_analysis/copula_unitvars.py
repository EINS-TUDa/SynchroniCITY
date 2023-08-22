import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

from readdata import readdata
import sys
sys.path.append("..")
from sync_model import distribution as dist


def calc_unitvars(data):
    ldists = {}
    data_ul = pd.DataFrame(index=data.index)  # unit variables for lognormal
    data_uln = pd.DataFrame(index=data.index)  # normal
    data_ue = pd.DataFrame(index=data.index)  # unit variables for empirical
    data_ue8 = pd.DataFrame(index=data.index)  # copula limit 0.1-0.9

    for hh in data:
        h = data[hh]
        ldists[hh] = dist.lognormal_EV(h.mean(), h.var())
        data_ul[hh] = ldists[hh].cdf(h)
        data_uln[hh] = stats.norm.ppf(data_ul[hh])
        hsort = np.sort(h)
        data_ue[hh] = (np.searchsorted(hsort, h) + 1) / (len(h) + 1)
        data_ue8[hh] = data_ue[hh] * 0.8 + 0.1

    return data_ue, data_ue8, data_ul, ldists, data_uln


def plot_hist_dist(data, dists, hh):
    h = data[hh]
    d = dists[hh]
    h.plot.hist(bins=2000, log=True, density=True)
    x = np.linspace(0, h.max(), 2000)
    plt.plot(x, d.pdf(x), label=f"Lognormal(E={h.mean():.1f},V={h.std():.1f}²)")
    plt.ylim(1e-6)
    plt.legend()


def plot_data_all_at(data, year, month, day, hour, minute=0, second=0):
    plt.figure()
    date = datetime(year, month, day, hour, minute, second)
    data.loc[date].plot(kind="bar")
    plt.ylabel("Power [unit vars]")
    plt.title(f"Power at {date}")


if __name__ == "__main__":
    data, psum = readdata("1min", 2020, with_hp=False)
    # data, psum = readdata("10s",with_hp=False,dateend=24 * 60 * 60)

    data_ue, data_ue8, data_ul, ldists, data_uln = calc_unitvars(data)

    data_ul["HH3"].hist(bins=200)
    plt.show()
    plot_hist_dist(data, ldists, "HH7")
    plt.show()
    plot_data_all_at(data_ue, 2020, 1, 11, 18)
    plt.show()

    corr = data_uln.corr()
