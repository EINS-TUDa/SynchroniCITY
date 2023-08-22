import matplotlib.pyplot as plt
import numpy as np

from syncmodel import SyncModel

prop_cycle = plt.rcParams['axes.prop_cycle']
plt_colors = prop_cycle.by_key()['color']

# ns = [10,20,50,100,200,500,1000]
ns = [10, 100, 1000]
ylimits = [2, 4, 8, 16]
prbs = {}

for n in ns:
    s = SyncModel(n, 0.01, sample_size=5e6)
    prbs[n] = dict((y, s.calc_prob_exceed(None, y, True)) for y in ylimits)

fig, ax = plt.subplots()
for i, y in enumerate(ylimits):
    plt.plot(s.rs[:-1], prbs[100][y], label=f"P(Sum > {f'{y:g} Mean'})", color=plt_colors[i], )
    plt.plot(s.rs, prbs[10][y], color=plt_colors[i], linestyle="--")
    plt.plot(s.rs, prbs[1000][y], color=plt_colors[i], linestyle=":")
plt.xlabel("Synchronization")
plt.xlim(0, 1)
plt.grid()
ax.set_yscale('log')
# 2nd legend for linestyle with black
lines = ax.get_lines()
leg1 = plt.legend([lines[0], lines[1], lines[2]], [f"n={n}" for n in [100, 10, 1000]], loc="upper right")
plt.legend()
plt.gca().add_artist(leg1)
plt.ylim(1e-5)
