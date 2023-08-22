from syncmodel import SyncModel
import matplotlib.pyplot as plt
import numpy as np

s = SyncModel(100, 0.01, sample_size=1e6, corr_mat_type="SCALEFREE", corr_params={"sf_m": 99})


def plot_pexc_m(ylimit, log=True, normalized=True, sample_size=1e5, save=True):
    fig, ax = plt.subplots()
    for m in [1, 2, 3, 4, 5, 10, 20, 50, 99]:  # np.linspace(0, 1, 10 + 1):
        s = SyncModel(100, 0.01, sample_size=sample_size, corr_mat_type="SCALEFREE", corr_params={
            "sf_m": m,  # m  edges to connect
        }, savefile_fs=save)
        pexc = s.calc_prob_exceed(None, ylimit, normalized=True)
        plt.plot(s.rs, pexc, label=f"m={m:g}")
    plt.xlabel("Synchronization")
    plt.xlim(0, 1)
    plt.ylabel(f"P(Sum > {f'{ylimit:g} Mean' if normalized else f'{ylimit:.0f} kW'})")
    plt.legend()
    plt.grid()
    if log:
        ax.set_yscale('log')


ylimit = 4
plot_pexc_m(ylimit, sample_size=1e5, save=False)
