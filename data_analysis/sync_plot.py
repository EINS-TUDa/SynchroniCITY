import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
from datetime import date, datetime, timedelta, time

sync = pickle.load(open(f"sync_results/sync_2019_cly_cl8", "rb"))
# sync = pickle.load(open(f"sync_results/sync_2019_gaus_cl8", "rb"))
# sync = pickle.load(open(f"sync_results/sync_2019_gaus_ln", "rb"))
sync2019 = pickle.load(open("sync_results/sync_2019_cly_cl8", "rb"))
sync2020 = pickle.load(open("sync_results/sync_2020_cly_cl8", "rb"))

sync.plot()


def plot_load_and_sync(sync, start, end):
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(psum[start:end], label="Sum")
    ax[0].legend()
    ax[0].ylabel("Load [W]")
    for c in data:
        ax[1].plot(data[c][start:end], label=c)
    ax[1].ylabel("Load [W]")
    ax[2].plot(sync[start:end], label="Sync", color="tab:orange")
    plt.legend()


plot_load_and_sync(sync, 60 * 24 * 40, 60 * 24 * 41)

sync_gr_hr = sync.groupby(sync.index.hour)
sync_gr_hr.mean().plot()
plt.fill_between(sync_gr_hr.mean().index, sync_gr_hr.mean() - sync_gr_hr.std(), sync_gr_hr.mean() + sync_gr_hr.std(),
                 alpha=0.2)

sync_gr_wy = sync.groupby(sync.index.weekofyear)
sync_gr_wy.mean().plot()
plt.fill_between(sync_gr_wy.mean().index, sync_gr_wy.mean() - sync_gr_wy.std(), sync_gr_wy.mean() + sync_gr_wy.std(),
                 alpha=0.2)

sync_gr_dayofyear = sync.groupby(sync.index.dayofyear)
sync_gr_dayofyear.mean().plot()
plt.fill_between(sync_gr_dayofyear.mean().index, sync_gr_dayofyear.mean() - sync_gr_dayofyear.std(),
                 sync_gr_dayofyear.mean() + sync_gr_dayofyear.std(), alpha=0.2)


def compare_2019_2020(sync2019=sync2019, sync2020=sync2020):
    sync2019.groupby(sync2019.index.dayofyear).mean().plot()
    sync2020.groupby(sync2020.index.dayofyear).mean().plot()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.xlim(0, 365)
    plt.legend(["2019", "2020"])
    plt.ylabel("Synchronization")
    plt.xlabel("")


def comp1920_hrs():
    sync_filt19 = sync2019[sync2019.index.hour.isin(range(12, 22))]
    sync_filt20 = sync2020[sync2020.index.hour.isin(range(12, 22))]
    compare_2019_2020(sync_filt19, sync_filt20)


def plot_sync_year():
    # convert groupby object to dataframe 2D
    sync_time_days = pd.DataFrame(
        dict((c, list(sync[ind])) for c, ind in sync.groupby(sync.index.time).indices.items()))

    im = plt.imshow(sync_time_days, origin="lower", extent=[0, 24, 0, 365], aspect="auto", cmap="rainbow")
    plt.colorbar(im)
    plt.xticks(np.arange(0, 24, 2))
    plt.xlabel("Hours of day")
    plt.yticks([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
               ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)


def plot_sync_profile(sync, days, label=None, quantile=False, **args):
    sync_filt = sync[sync.index.dayofweek.isin(days)]  # monday:0
    sync_gr_time = sync_filt.groupby(sync_filt.index.time)
    sync_time_days = pd.DataFrame(dict((c, list(sync_filt[ind])) for c, ind in sync_gr_time.indices.items()))

    mean = sync_time_days.mean()
    mean.plot(label=label, **args)
    # improve x axis
    hrs = np.arange(0, 24, 2)
    plt.xticks([time(h, 0) for h in hrs], hrs)
    plt.xlabel("Hours of day")
    plt.xlim(time(0, 0), time(23, 59))
    plt.ylabel("Synchronization")
    if quantile:
        q = np.quantile(sync_time_days, [.1, .9], axis=0)
        plt.fill_between(mean.index, q[0], q[1], alpha=0.2)

    # im = plt.imshow(sync_time_days - mean, origin="lower", extent=[0, 24, 0, 365], aspect="auto", cmap="RdYlBu_r")


def plot_sync_profile_all():
    plot_sync_profile(sync, [0, 1, 2, 3, 4, 5, 6], "All", True)
    plot_sync_profile(sync, [0, 1, 2, 3, 4], "Weekday")
    plot_sync_profile(sync, [5], "Saturday")
    plot_sync_profile(sync, [6], "Sunday", color="brown")
    plt.legend(loc="lower left")


def syp_2020_compare(days):
    plt.figure()
    plot_sync_profile(sync2019, days, "2019", color="tab:blue")
    plot_sync_profile(sync2020, days, "2020", color="tab:orange")
    plt.grid()
    plt.legend()


def syp_2020_compare_all():
    syp_2020_compare([0, 1, 2, 3, 4, 5, 6])
    syp_2020_compare([0, 1, 2, 3, 4])
    syp_2020_compare([5])
    syp_2020_compare([6])


def make_uniform(data):
    d = np.asarray(data)
    dsort = np.sort(d.reshape(-1))
    return (np.searchsorted(dsort, d) + 1) / (len(dsort) + 1)


def make_uniform_linear(data):
    d = np.asarray(data)
    d2 = d.reshape(-1)
    dmin, dmax = d2.min(), d2.max()
    return (d - dmin) / (dmax - dmin)


def plot_psum_sync_alpha(alphamin=0.2):
    plt.figure()
    psum_time = psum.groupby(psum.index.time)
    psum_time_df = pd.DataFrame(dict((c, list(psum[ind])) for c, ind in psum_time.indices.items()))
    im = plt.imshow(psum_time_df, origin="lower", extent=[0, 24, 0, 365], aspect="auto", cmap="turbo")
    plt.colorbar(im)

    plt.figure()
    uni_psum = make_uniform(psum_time_df)
    # uni_psum=make_uniform_linear(psum_time_df)
    im = plt.imshow(sync_time_days, origin="lower", extent=[0, 24, 0, 365], aspect="auto",
                    cmap="rainbow", alpha=alphamin + (1 - alphamin) * uni_psum)
    im.axes.patch.set_facecolor('black')


def plot_2d_colormap(alphamin=0.2):  # plot 2nd image for 2d colormap
    plt.figure()
    pmin, pmax = psum.min(), psum.max()
    smin, smax = sync.min(), sync.max()
    im = plt.imshow([np.full(500, i) for i in np.linspace(smin, smax, 500)],
                    origin="lower", extent=[pmin, pmax, smin, smax], aspect="auto", cmap="rainbow",
                    alpha=[alphamin + (1 - alphamin) * make_uniform(np.linspace(pmin, pmax, 500)) for _ in range(500)])
    im.axes.patch.set_facecolor('black')
