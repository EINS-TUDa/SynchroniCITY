import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import LogNorm, Normalize

from readdata import readdata

data, psum = readdata("1min", 2019, with_hp=False, unit=1000)

sync2020 = pickle.load(open("sync_results/sync_2020_cly_cl8", "rb"))
sync2019 = pickle.load(open("sync_results/sync_2019_cly_cl8", "rb"))
sync = sync2019

sync.plot.hist(bins=200, density=True, )


def plot_syncvssum():
    sync_bins = pd.cut(sync, np.linspace(0, 1, 11))
    bins = sync_bins.dtype.categories
    for bin in bins[:-2]:
        synci = sync[sync_bins == bin].index
        print(bin)
        p = psum[synci]
        t = synci.time
        # t = t.hour * 3600 + t.minute * 60 + t.second
        print(f"Samples: {p.count()}, Mean: {p.mean()}, Time: {synci.array.hour.mean()}")
        p.plot.hist(bins=200, density=True, histtype="step", label=bin)
        # t.plot.hist(bins=24, density=True, histtype="step", label=bin)
    plt.legend()


def plot_sumvssync():
    p_bins = pd.cut(psum, 10)
    pbins = p_bins.dtype.categories
    for bin in pbins:
        psumi = psum[p_bins == bin].index
        print(bin)
        s = sync[psumi]
        t = psumi.time
        print(f"Samples: {s.count()}, Mean: {s.mean()}")
        s.plot.hist(bins=200, density=True, histtype="step", label=bin)
    plt.legend()


def heatmap():
    ps = pd.DataFrame({"psum": psum, "sync": sync})
    # plt.scatter(ps.sync, ps.psum, s=0.05,)
    ps["psum_bin"] = pd.cut(ps.psum, 200)
    ps["sync_bin"] = pd.cut(ps.sync, np.linspace(0, 1, 200 + 1))
    psg = ps.groupby(["psum_bin", "sync_bin"]).count()["psum"]
    psg = psg.unstack()
    plt.imshow(psg, cmap=None, origin="lower",
               extent=[psg.axes[1][0].left, psg.axes[1][-1].right, psg.axes[0][0].left, psg.axes[0][-1].right, ],
               aspect="auto", norm=LogNorm(clip=True), interpolation="nearest")
    plt.xlabel("Synchronization")
    plt.ylabel("Total Power (kW)")
    plt.ylim(None, 31)


def compare_sync_sum_flucs():
    sync1 = sync / sync.std()
    psum1 = psum / psum.std()
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(sync1)
    ax[1].plot(psum1)
    print(f"Sync diff std: {sync1.diff().std()}")
    print(f"Sum diff std: {psum1.diff().std()}")
