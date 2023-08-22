import pandas as pd
import matplotlib.pyplot as plt
import os

pathtodata = (os.path.dirname(__file__) + "/data").lstrip("/")


# reads the data from the hdf5 file and returns a dataframe with the data of all households with full data availability
def readdata(timeres, year=2019, with_hp=False, datestart=0, dateend=None, unit=1):  # dateend in seconds
    file = f"{pathtodata}/{year}_data_{timeres}.hdf5"
    samplerate = {
        "15min": 60 * 15,
        "1min": 60,
        "10s": 10,
    }[timeres]
    exclude = {13, 15, 26, 33}  # PV
    exclude |= {
        2019: {24, 6, 17, 37, 40, 25, 31, 34},  # availability<99% ; [34] unusual behavior
        2020: {24, 25, 6, 8, 11, 17, 31, 35, 10, 23},  # availability<99%
    }[year]
    data = {}
    for i in range(3, 41):
        if i in exclude:
            continue
        d = pd.read_hdf(file, f"NO_PV/SFH{i}/HOUSEHOLD/table")
        if dateend:
            d = d[datestart:int(dateend / samplerate)]
        d.index = pd.to_datetime(d["index"], unit='s')
        hh = d["P_TOT"]
        if with_hp:
            hp = pd.read_hdf(file, f"NO_PV/SFH{i}/HEATPUMP/table")["P_TOT"]
            if dateend:
                hp = hp[datestart:int(dateend / samplerate)]
            hp.index = d.index
            p = hh + hp
        else:
            p = hh
        data[f"HH{i}"] = p

    df = pd.DataFrame(data)
    psum = df.sum(axis=1)
    return df / unit, psum / unit


def hh_hist_comp(timeres, i):
    file = f"data/2019_data_{timeres}.hdf5"
    d = pd.read_hdf(file, f"NO_PV/SFH{i}/HOUSEHOLD/table")
    d.index = pd.to_datetime(d["index"], unit='s')
    hh = d["P_TOT"]
    hp = pd.read_hdf(file, f"NO_PV/SFH{i}/HEATPUMP/table")["P_TOT"]
    hp.index = d.index
    p = hh + hp
    p.plot.hist(bins=500, log=True)
    hp.plot.hist(bins=500, log=True)
    hh.plot.hist(bins=500, log=True)
    plt.legend(["Total", "Heatpump", "Household"])
    plt.xlabel("Power Consumption [W]")
    plt.title(f"Household {i}")


if __name__ == "__main__":
    data, psum = readdata("1min", 2019, with_hp=False)

    data["HH7"].plot()
    plt.show()
    hh_hist_comp("1min", 7)
    plt.show()
