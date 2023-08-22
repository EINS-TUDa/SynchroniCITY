from slp_vdew import winter
from syncmodel import SyncModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

s = SyncModel(100, 0.01, sample_size=1e6)
s = SyncModel(100, 0.001, sample_size=1e6)
s = SyncModel(100, 0.001, sample_size=1e5)

# winter.plot()
wt = winter["Werktag"]
slp = wt / wt.mean() * s.mean

t = [d.hour + d.minute / 60 for d in slp.index]
s.plot_totalpower_time(t, slp)
s.plot_totalpower_time_and_most_likely_r(t, slp, quantiles=[0.2, 0.8])

plt.subplots(2, 1, sharex=True)
slp.plot()
r_ml = s.get_most_likely_r_at_y(slp.values)
plt.subplot(2, 1, 1)
pd.Series(r_ml, index=slp.index).plot()
