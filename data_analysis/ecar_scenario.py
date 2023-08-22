import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.append("..")
sys.path.append("../sync_model")
from sync_model.syncmodel import SyncModel
from readdata import readdata

data, psum = readdata("1min", 2019, with_hp=False, unit=1000)

prosumers = [{
    "pdftype": "LOGNORMAL",
    "mean": data[hh].mean(),
    "std": data[hh].std(),
    "name": hh,
} for hh in data]
prosumers_ecar = [{
    "pdftype": "LOGNORMAL",
    "mean": data[hh].mean(),
    "std": data[hh].std(),
    "name": hh,
    "ecar": True,
} for hh in data]
prosumers_ecar50 = [{
    "pdftype": "LOGNORMAL",
    "mean": data[hh].mean(),
    "std": data[hh].std(),
    "name": hh,
    "ecar": int(hh.lstrip("HH")) < 20,
} for hh in data]
prosumers_ecar15 = [{
    "pdftype": "LOGNORMAL",
    "mean": data[hh].mean(),
    "std": data[hh].std(),
    "name": hh,
    "ecar": int(hh.lstrip("HH")) < 8,
} for hh in data]

# s = SyncModel(prosumers, 0.01, sample_size=5e6, copula_type="CLAYTON", copula_limit=[.1, .9])
# se = SyncModel(prosumers_ecar, 0.01, sample_size=5e6, copula_type="CLAYTON", copula_limit=[.1, .9])
s = SyncModel(prosumers, 0.01, sample_size=5e6, copula_type="CLAYTON")
se = SyncModel(prosumers_ecar, 0.01, sample_size=5e6, copula_type="CLAYTON")
se50 = SyncModel(prosumers_ecar50, 0.01, sample_size=5e6, copula_type="CLAYTON")
se15 = SyncModel(prosumers_ecar15, 0.01, sample_size=5e6, copula_type="CLAYTON")

s.plot_total_densities(mcshist=True)
se.plot_total_densities(mcshist=True)
s.plot_prob_exceed()
se.plot_prob_exceed()
se50.plot_prob_exceed()
se15.plot_prob_exceed()

fig, ax = plt.subplots()
y = 8
plt.plot(s.rs, s.calc_prob_exceed(None, y, True), label="No EV", color="tab:blue")
plt.plot(se15.rs, se15.calc_prob_exceed(None, y, True), label="15% EV", color="tab:green")
plt.plot(se50.rs, se50.calc_prob_exceed(None, y, True), label="50% EV", color="tab:orange")
plt.plot(se.rs, se.calc_prob_exceed(None, y, True), label="100% EV", color="tab:red")
plt.xlabel("Synchronization")
plt.xlim(0, 1)
plt.ylim(6e-6)
plt.legend()
plt.grid()
ax.set_yscale('log')
