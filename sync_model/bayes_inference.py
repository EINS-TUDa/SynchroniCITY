import matplotlib.pyplot as plt
import distribution as dist
from syncmodel import SyncModel

s = SyncModel(100, 0.01, sample_size=1e6)
s = SyncModel(100, 0.01, sample_size=5e6)
s = SyncModel(100, 0.001, sample_size=1e5)
s.plot_densities_and_most_likely_r()
s.plot_total_densities(rs_fac=10, colorbar=True, ymax=150)
s.plot_densities_and_pexceed(rs_fac=10)

pr = dist.beta(1, 1)
pr = dist.beta(2, 10)
pr = dist.beta(1, 8)
pr = dist.beta(1, 5)
pr = dist.beta(1, 10, plot=True)
s.plot_densities_and_most_likely_r(prior=pr)

yss = [0.2, 0.5, 0.8, 1, 1.4, 2, 3]
yss = [0.2, 0.5, 1, 2, 3]
yss = [0.1, 0.2, 0.25, 0.5, 0.8, 1, 1.25, 2, 4, 5, 10]
s.plot_pry(pr, yss, normalized=True)
plt.xlim(0, 0.8)

yss = [10, 25, 60, 100]
yss = [25, 50, 70, 100, 150]
s.plot_pry(pr, yss, normalized=False)

s.plot_total_densities()
prop_cycle = plt.rcParams['axes.prop_cycle']
plt_colors = prop_cycle.by_key()['color']
for i, y in enumerate(yss):
    plt.vlines(y, 0, s.fs[0].max(), linestyle="--", color=plt_colors[i])
