from syncmodel import SyncModel
import matplotlib.pyplot as plt
import numpy as np

s = SyncModel(100, 0.1, sample_size=1e6)
# s = SyncModel(100,0.1,sample_size=1e7)
# s = SyncModel(100,0.01,sample_size=2e7)
# s = SyncModel(100,0.01,sample_size=1e6)
# s = SyncModel(100,0.001,sample_size=1e5)

s.plot_total_densities(ymax=3, normalized=True)
s.plot_total_densities(colorbar=True, ymax=3, normalized=True)
s.plot_total_densities(colorbar=True, ymax=160)
s.plot_total_densities(rs_fac=10, ymax=150)
s.plot_densities_and_most_likely_r()
s.plot_densities_and_most_likely_r(rs_fac=10, quantiles=[0.2, 0.8])
s.plot_densities_and_pexceed(normalized=True, log=False)
s.plot_prob_exceed()
s.plot_sync_total_2d_density()

# plt.gca().set_yscale('log')

s = SyncModel({"n": 100, "pdftype": "LOGNORMAL", "ecars": 0.4}, 0.1, sample_size=1e6)
s = SyncModel({"n": 100, "pdftype": "LOGNORMAL", "ecars": 1}, 0.1, sample_size=1e6)
s = SyncModel({"n": 100, "pdftype": "LOGNORMAL", "ecars": 1}, 0.01, sample_size=1e6)
# s = SyncModel({"n": 100, "pdftype": "LOGNORMAL", "mean": 0.01, "ecars": 1}, 0.1, sample_size=1e6)
s.plot_total_densities(mcshist=True)

# s = SyncModel({"n": 100, "pdftype": "BETA"}, 0.01, sample_size=1e5)
# s = SyncModel({"n": 100, "pdftype": "BETA"}, 0.1, sample_size=1e6)
