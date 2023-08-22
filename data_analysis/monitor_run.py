import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sys

sys.path.append("..")
sys.path.append("../sync_model")
from sync_model.syncmodel import SyncModel
from sync_model.bayes_filter_syncmodel import BayesFilterSM, System, plot_prob_exceed, plot_filters
from readdata import readdata

data, psum = readdata("1min", 2019, with_hp=False, unit=1000)  # convert to kW

prosumers = [{
    "pdftype": "LOGNORMAL",
    "mean": data[hh].mean(),
    "std": data[hh].std(),
    "name": hh,
} for hh in data]
# prosumers = [{
#     "pdftype": "ECDF",
#     "data": data[hh],
#     "name": hh,
# } for hh in data]

# s = SyncModel(prosumers, 0.01, sample_size=1e6, copula_type="GAUSS")
# s = SyncModel(prosumers, 0.01, sample_size=1e6, copula_type="GAUSS", copula_limit=[.1, .9])
# s = SyncModel(prosumers, 0.01, sample_size=5e6, copula_type="CLAYTON")
s = SyncModel(prosumers, 0.01, sample_size=5e6, copula_type="CLAYTON", copula_limit=[.1, .9])

s.plot_total_densities()
s.plot_total_densities(mcshist=True)
s.plot_densities_and_most_likely_r()

psum.plot.hist(bins=500, density=True, alpha=0.4, color="gray")
plt.gca().set_yscale("log")
plt.ylim(1e-5)
plt.xlim(0, 100)

# compare probabilities of exceeding limits
s.plot_prob_exceed()
psum_exc = [sum(psum > psum.mean() * lim) / len(psum) for lim in [2, 4, 8, 16]]

# compare with ground truth sync
sync = pickle.load(open("sync_results/sync_2019_cly_cl8", "rb"))
# sync = pickle.load(open("sync_results/sync_2019_gaus_cl8", "rb"))
# sync = pickle.load(open("sync_results/sync_2019_gaus_ln", "rb"))
sync.plot.hist(bins=100, density=True, histtype="step")

sync_ml = pd.Series(s.get_most_likely_r_at_y(psum), index=psum.index)
sync_ml.plot.hist(bins=100, density=True)
counts, _ = np.histogram(sync_ml, bins=int(1 / s.r_res) + (1 if s.copula_type == "GAUSS" else -1), density=True)
prior = counts * s.r_res

# analyze error
sync_ml_err = sync_ml - sync
sync_ml_err.plot.hist(bins=100, density=True, histtype="step", grid=True)
plt.plot(psum, sync_ml_err, ".")
print(f"MSE: {(sync_ml_err ** 2).mean()}, STD: {sync_ml_err.std()}")

ylimits = [4]
bf = BayesFilterSM(s, System.NORMAL(len(s.rs), 0.02), name="Bayes Filter (0.02)", firstprior=prior)
bfs = []
bfs.append(bf)
bfs.append(BayesFilterSM(s, System.NORMAL(len(s.rs), 0.01), name="Bayes Filter (0.01)", firstprior=prior))


# bfs.append(BayesFilterSM(s, System.NORMAL_PLUS_BASE(len(s.rs), 0.01, .01), name="Bayes Filter (0.01+base1%)",
#                          firstprior=prior))
# bfs.append(BayesFilterSM(s, System.NORMAL_PLUS_BASE(len(s.rs), 0.01, .01), name="Bayes Filter (0.01+base1%) - Argmax",
#                          firstprior=prior, estimate_argmax=True))


def run_bfs(bfs, sidx, dur, plot_ml=False, ylimits=None):
    t = data.index[sidx:sidx + dur]
    y = psum[t]
    sync_gt = sync[t]
    for bf in (bfs if hasattr(bfs, "__iter__") else [bfs]):
        if ylimits:
            bf.track_limit_probs(ylimits, normalized=True)
        bf.all_meas(y)
    sync_est = pd.Series(bf.estimates, index=t)
    sync_err = sync_est - sync_gt
    print(f"MSE: {(sync_err ** 2).mean()}, STD: {sync_err.std()}")
    plot_filters(t, y, sync_gt, bfs, plot_y=True, syncm=s, plot_ml=plot_ml, rs_hlines=None, adjtimex=True)
    return t, sync_est


t, sync_est = run_bfs(BayesFilterSM(s, System.NORMAL(len(s.rs), 0.02), firstprior=prior), 60 * (24 * 103 + 3), 601)
t, sync_est = run_bfs(bfs, 60 * (24 * 82 + 3), 601)

plot_filters(t, y, sync_gt, bf)

sync_err.plot.hist(bins=100, density=True, grid=True)
# small points
plt.plot(psum, sync_err, ".", markersize=1)
# heat map of error
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "orange"])
plt.hist2d(psum, sync_err, bins=200, cmap=cmap)
plt.grid(True)
plt.xlabel("Total Power (kW)")
plt.ylabel("Synchronization Monitoring Error")
plt.xlim(None, 20)
