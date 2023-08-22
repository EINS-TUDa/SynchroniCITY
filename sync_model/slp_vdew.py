import pandas as pd
import xlrd
import datetime

data = pd.read_excel('SLP/Repräsentative Profile VDEW.xls', "H0", header=0)
name = data.columns[0]
# print(name)
data = data.drop(98)  # text
data = data.drop(97)  # drop 0:00:00
time = data[data.columns[0]][2:]
time.rename("Time", inplace=True)


def season(subdata):
    days = []
    for c in subdata.columns:
        days.append(subdata[c][1])
    return pd.DataFrame(subdata[2:].values, index=time, columns=days, dtype=float)


winter = season(data[data.columns[1:4]])

if __name__ == "__main__":
    wt = winter["Werktag"]
    wt.plot()

    wt.mean()
