import pandas as pd
import pickle
import matplotlib.pyplot as plt

syncs = {
    "gaus_emp": pickle.load(open(f"sync_results/sync_2019_gaus_emp", "rb")),
    "cly_emp": pickle.load(open(f"sync_results/sync_2019_cly_emp", "rb")),
    "frk_emp": pickle.load(open(f"sync_results/sync_2019_frk_emp", "rb")),
    "gum_emp": pickle.load(open(f"sync_results/sync_2019_gum_emp", "rb")),
    "gaus_cl8": pickle.load(open(f"sync_results/sync_2019_gaus_cl8", "rb")),
    "cly_cl8": pickle.load(open(f"sync_results/sync_2019_cly_cl8", "rb")),
    "frk_cl8": pickle.load(open(f"sync_results/sync_2019_frk_cl8", "rb")),
    "gum_cl8": pickle.load(open(f"sync_results/sync_2019_gum_cl8", "rb")),
    "gaus_ln": pickle.load(open(f"sync_results/sync_2019_gaus_ln", "rb")),
    "cly_ln": pickle.load(open(f"sync_results/sync_2019_cly_ln", "rb")),
    "frk_ln": pickle.load(open(f"sync_results/sync_2019_frk_ln", "rb")),
    "gum_ln": pickle.load(open(f"sync_results/sync_2019_gum_ln", "rb")),
}

syncs["gaus_ln"].plot(label="Gaussian Logn")
syncs["gaus_emp"].plot(label="Gaussian Emp")
syncs["gaus_cl8"].plot(label="Gaussian Emp .1-.9")
syncs["gum_ln"].plot(label="Gumbel Logn")
syncs["gum_emp"].plot(label="Gumbel Emp")
syncs["gum_cl8"].plot(label="Gumbel Emp .1-.9")
syncs["cly_ln"].plot(label="Clayton Logn")
syncs["cly_emp"].plot(label="Clayton Emp")
syncs["cly_cl8"].plot(label="Clayton Emp .1-.9")
syncs["frk_ln"].plot(label="Frank Logn")
syncs["frk_emp"].plot(label="Frank Emp")
syncs["frk_cl8"].plot(label="Frank Emp .1-.9")
plt.legend(loc="upper right")

plt.hist(syncs["gaus_ln"], bins=500, histtype="step", density=True, label="Gaussian Logn")
plt.hist(syncs["gaus_emp"], bins=500, histtype="step", density=True, label="Gaussian Emp")
plt.hist(syncs["gaus_cl8"], bins=500, histtype="step", density=True, label="Gaussian Emp .1-.9")
plt.hist(syncs["gum_ln"], bins=500, histtype="step", density=True, label="Gumbel Logn")
plt.hist(syncs["gum_emp"], bins=500, histtype="step", density=True, label="Gumbel Emp")
plt.hist(syncs["gum_cl8"], bins=500, histtype="step", density=True, label="Gumbel Emp .1-.9")
plt.hist(syncs["cly_ln"], bins=500, histtype="step", density=True, label="Clayton Logn")
plt.hist(syncs["cly_emp"], bins=500, histtype="step", density=True, label="Clayton Emp")
plt.hist(syncs["cly_cl8"], bins=500, histtype="step", density=True, label="Clayton Emp .1-.9")
plt.hist(syncs["frk_ln"], bins=500, histtype="step", density=True, label="Frank Logn")
plt.hist(syncs["frk_emp"], bins=500, histtype="step", density=True, label="Frank Emp")
plt.hist(syncs["frk_cl8"], bins=500, histtype="step", density=True, label="Frank Emp .1-.9")
plt.legend(loc="upper right")

syncs["gaus_ln"].diff().plot(label="Gaussian Logn")
syncs["cly_ln"].diff().plot(label="Clayton Logn")
syncs["gaus_ln"].diff().plot.hist(bins=200, histtype="step", label="Gaussian Logn")
syncs["cly_ln"].diff().plot.hist(bins=200, histtype="step", label="Clayton Logn")

syncinfo = []
for cop in syncs:
    syncinfo.append({
        # "name": cop,
        # "count": syncs[cop].count(),
        "mean": syncs[cop].mean(),
        "std": syncs[cop].std(),
        # "min": syncs[cop].min(),
        "1%": syncs[cop].quantile(0.01),
        "10%": syncs[cop].quantile(0.1),
        # "25%": syncs[cop].quantile(0.25),
        "median": syncs[cop].median(),
        # "75%": syncs[cop].quantile(0.75),
        "90%": syncs[cop].quantile(0.9),
        "99%": syncs[cop].quantile(0.99),
        "max": syncs[cop].max(),
        # "IQR": syncs[cop].quantile(0.75) - syncs[cop].quantile(0.25),
        "is NaN [%]": syncs[cop].isna().sum() / syncs[cop].size * 100,
        "is 0 [%]": syncs[cop].lt(0.005).sum() / syncs[cop].size * 100,
        # "autocorr": syncs[cop].autocorr(),
        # "AAC": syncs[cop].diff().abs().mean(), # Average Absolute Change
        # "ACskew": syncs[cop].diff().abs().skew(), # Skewness of Change
        "diff std": syncs[cop].diff().std(),
    })
syncinfo = pd.DataFrame(syncinfo, index=syncs.keys())

sync2019 = pickle.load(open("sync_results/sync_2019_cly_cl8", "rb"))
sync2020 = pickle.load(open("sync_results/sync_2020_cly_cl8", "rb"))

sync1920info = []
for s in (sync2019, sync2020):
    sync1920info.append({
        "mean": s.mean(),
        "std": s.std(),
        "1%": s.quantile(0.01),
        "10%": s.quantile(0.1),
        "median": s.median(),
        "90%": s.quantile(0.9),
        "99%": s.quantile(0.99),
        "max": s.max(),
        "is 0 [%]": s.lt(0.005).sum() / s.size * 100,
        "diff std": s.diff().std(),
    })
sync1920info = pd.DataFrame(sync1920info, index=["2019", "2020"])
