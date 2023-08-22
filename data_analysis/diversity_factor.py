from readdata import readdata, hh_hist_comp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta, time

data, psum = readdata("1min", with_hp=False, unit=1000)

psum.plot()
psum.plot.hist(bins=2000, log=True)

coinc_fac_all = psum.max() / sum(data[c].max() for c in data)

coinc_intv = 60
coinc_fac = pd.Series(index=data.index, dtype="float64")
for date in range(0, len(data), int(coinc_intv)):
    start, end = date, date + int(coinc_intv)
    coinc_fac[start:end] = psum[start:end].max() / sum(data[c][start:end].max() for c in data)

coinc_fac.plot(label=f"CF {coinc_intv} min")
coinc_fac.plot.hist(bins=100)

day_samples = int(24 * 3600 / (data.index[1] - data.index[0]).seconds)
cf_day = pd.DataFrame(index=data.index[:day_samples] - data.index[0], dtype="float64", columns=range(365))
for time in range(0, day_samples):
    cf_day.iloc[time] = coinc_fac[range(time, len(data), day_samples)]

cf_day.mean(axis=1).plot(label=f"CF {coinc_intv} min")
im = plt.imshow(cf_day.T, origin="lower", extent=[0, 24, 1, 365], interpolation=None, aspect="auto")
plt.colorbar(im)


def plot_load_and_divf(day, sync=None):
    start, end = day_samples * day, day_samples * (day + 1)
    date = psum.index[start]
    print(f"{date.strftime('%Y-%m-%d')} ({date.strftime('%A')})")
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(psum[start:end])
    ax[0].set_ylabel("Total Power (kW)")
    ax[0].grid()
    for c in data:
        ax[1].plot(data[c][start:end], label=c)
    ax[1].set_ylim(0)
    ax[1].grid()
    ax[1].set_ylabel("Individual\nPower (kW)")
    ax[2].plot(coinc_fac[start:end], label="Coincidence Factor")
    if sync is not None:
        ax[2].plot(sync[start:end], label="Synchronization")
    ax[2].legend(loc="lower right")
    ax[2].ylim(0)
    ax[2].grid()

    hrs = np.arange(0, 24, 2)
    plt.xticks([datetime(date.year, date.month, date.day, h, 0, 0) for h in hrs], hrs)
    plt.xlabel("Hours of day")
    plt.xlim(date, date + timedelta(days=1))
