import matplotlib.colors
import matplotlib.pyplot as plt
import math

default_cmap = "turbo"
cmap = plt.colormaps.get(default_cmap)


def get(y):  # y normalized
    # if y<1:
    #     return cmap(y/2)
    # else:
    #     return cmap(1-1/2/y)
    return cmap(2 / math.pi * math.atan(y))


def create_cmap(fsx, mean):
    # colors = np.vectorize(totalpower_cmap.get)(self.fsx / self.mean)
    colors = [get(x / mean) for x in fsx]
    norm = plt.Normalize(min(fsx), max(fsx))
    tuples = list(zip(map(norm, fsx), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    return cmap, norm
