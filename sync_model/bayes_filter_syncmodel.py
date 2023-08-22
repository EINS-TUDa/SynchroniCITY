from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from syncmodel import SyncModel
from bayes_filter import BayesFilter, System


# A specialized class of BayesFilter for handling the sync model

class BayesFilterSM(BayesFilter):

    def __init__(self, syncmodel: SyncModel, system=System.STATIC, firstprior=None, name=None, **kwargs):
        self.syncmodel = syncmodel
        state_values = syncmodel.rs
        likelihood = syncmodel.get_likelihood_at_y
        super().__init__(state_values, likelihood, system, firstprior, name=name, **kwargs)
        self.state_label = "Synchronization"
        self.limit_probs = []
        self.exceed_prob = {}

    def track_limit_probs(self, ylimits, normalized=True):
        if type(ylimits) in {int, float}:  # if input = number
            ylimits = [ylimits]
        self.limit_probs.extend(ylimits)
        for limit in ylimits:
            self.exceed_prob[limit] = []
        self.ylimit_normalized = normalized

    def new_meas(self, measurement):
        super().new_meas(measurement)
        for limit in self.limit_probs:
            self.exceed_prob[limit].append(
                self.syncmodel.calc_prob_exceed_restimate(self.prior, limit, self.ylimit_normalized))


def plot_filters(t, y, r, filters, plot_y=False, syncm=None, plot_p=False, gt_label="Ground Truth",
                 exceed_prob_gt=None, plot_fill_quantiles=True, plot_ml=False, rs_hlines=0.8, adjtimex=False):
    if not hasattr(filters, "__iter__"):
        filters = [filters]
    if plot_y and plot_p:
        fig, (ax, ax_y, ax_p) = plt.subplots(3, 1, sharex=True)
    elif plot_y and not plot_p:
        fig, (ax, ax_y) = plt.subplots(2, 1, sharex=True)
    elif not plot_y and plot_p:
        fig, (ax, ax_p) = plt.subplots(2, 1, sharex=True)
        ax_p.set_xlabel("Time")
    else:
        fig, ax = plt.subplots()
        ax.set_xlabel("Time")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("Synchronization")
    if plot_ml:
        ax.plot(t, filters[0].syncmodel.get_most_likely_r_at_y(y), label="ML sync", color="gray")
    ax.plot(t, r, label=gt_label, color="k")
    for f in filters:
        ax.plot(t, f.estimates, label=str(f))
        if plot_fill_quantiles:
            ax.fill_between(t, f.estimates_quantiles[0.1], f.estimates_quantiles[0.9], alpha=0.2)
    ax.legend()
    ax.grid()
    ax.set_ylim(0)
    if plot_y:
        syncm.plot_totalpower_time(t, y, ax=ax_y, rsmax=rs_hlines, adjtimex=adjtimex)
    if plot_p:
        plot_prob_exceed(t, filters, exceed_prob_gt, list(exceed_prob_gt.keys()), ax=ax_p)
    plt.show()


def plot_prob_exceed(t, filters, exceed_prob_gt, limits=[], ax=None, normalized=None):
    if type(filters) is BayesFilterSM:
        filters = [filters]
    if normalized is None:
        normalized = filters[0].ylimit_normalized
    if ax is None:
        fig, ax = plt.subplots()
    for f in filters:
        for y in f.limit_probs:
            ax.plot(t, f.exceed_prob[y], label=str(f) if len(f.limit_probs) == 1 else f"{f} - Lim: {y}")
    for y, vals in exceed_prob_gt.items():
        ax.plot(t, vals, label="Ground Truth" if len(exceed_prob_gt) == 1 else f"Ground Truth - Lim: {y}", color="k")
    ax.set_ylabel(f"P(Sum > {f'{limits[0]:g} Mean' if normalized else f'{limits[0]:.0f} kW'})"
                  if len(limits) == 1 else "P(Sum > Limit)")
    ax.set_yscale('log')
    ax.set_ylim(1e-10, 1)
    ax.legend()
    ax.grid()
