from syncmodel import SyncModel
import matplotlib.pyplot as plt
import numpy as np

s = SyncModel(100, 0.1, sample_size=1e6, corr_mat_type="SMALLWORLD")
s = SyncModel(100, 0.01, sample_size=1e6, corr_mat_type="SMALLWORLD")


def plot_pexc_k(ylimit, log=True, normalized=True, sample_size=1e5, save=True):
    fig, ax = plt.subplots()
    for k in np.linspace(0.05, 1, 20):
        s = SyncModel(100, 0.01, sample_size=sample_size, corr_mat_type="SMALLWORLD", corr_params={
            "sw_kn": k,  # k mean degree, normalized to n
            "sw_p": 0.1,  # p rewire probability of SW net
        }, savefile_fs=save)
        pexc = s.calc_prob_exceed(None, ylimit, normalized=True)
        plt.plot(s.rs, pexc, label=f"k/n={k:g} ({int(k * s.n_prosum)})")
    plt.xlabel("Synchronization")
    plt.xlim(0, 1)
    plt.ylabel(f"P(Sum > {f'{ylimit:g} Mean' if normalized else f'{ylimit:.0f} kW'})")
    plt.legend()
    plt.grid()
    if log:
        ax.set_yscale('log')


def plot_pexc_p(ylimit, log=True, normalized=True, sample_size=1e5, save=True):
    fig, ax = plt.subplots()
    for p in np.linspace(0, 1, 10 + 1):
        s = SyncModel(100, 0.01, sample_size=sample_size, corr_mat_type="SMALLWORLD", corr_params={
            "sw_kn": 0.05,  # k mean degree, normalized to n
            "sw_p": p,  # p rewire probability of SW net
        }, savefile_fs=save)
        pexc = s.calc_prob_exceed(None, ylimit, normalized=True)
        plt.plot(s.rs, pexc, label=f"p={p:g}")
    plt.xlabel("Synchronization")
    plt.xlim(0, 1)
    plt.ylabel(f"P(Sum > {f'{ylimit:g} Mean' if normalized else f'{ylimit:.0f} kW'})")
    plt.legend()
    plt.grid()
    if log:
        ax.set_yscale('log')


ylimit = 4
plot_pexc_k(ylimit, sample_size=1e6)
plot_pexc_p(ylimit, sample_size=1e6)
