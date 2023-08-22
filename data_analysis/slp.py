import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from readdata import readdata, hh_hist_comp

data, psum = readdata("1min", with_hp=False)

day_samples = int(24 * 3600 / (data.index[1] - data.index[0]).seconds)

gr_hour = psum.groupby(psum.index.hour)
gr_hour.mean().plot()

gr_day = psum.groupby(psum.index.dayofyear)
gr_day.mean().plot()
plt.fill_between(gr_day.mean().index, gr_day.mean() - gr_day.std(), gr_day.mean() + gr_day.std(), alpha=0.2)

gr_weekyear = psum.groupby(psum.index.weekofyear)
gr_weekyear.mean().plot()
plt.fill_between(gr_weekyear.mean().index, gr_weekyear.mean() - gr_weekyear.std(),
                 gr_weekyear.mean() + gr_weekyear.std(), alpha=0.2)

gr_month = psum.groupby(psum.index.month)
gr_month.mean().plot()
plt.fill_between(gr_month.mean().index, gr_month.mean() - gr_month.std(), gr_month.mean() + gr_month.std(), alpha=0.2)

gr_time = psum.groupby(psum.index.time)
ps_filt = psum[psum.index.dayofweek == 6]
gr_time = ps_filt.groupby(ps_filt.index.time)
gr_time.mean().plot()
plt.fill_between(gr_time.mean().index, gr_time.mean() - gr_time.std(), gr_time.mean() + gr_time.std(), alpha=0.2)
plt.ylabel("Total Power [W]")

gr = psum.groupby([psum.index.month, psum.index.dayofweek, psum.index.hour])
