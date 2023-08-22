import math
import numpy as np


class ECDF:

    def __init__(self, data):
        self.values = np.sort(data)

    def max(self):
        return self.values[-1]

    def min(self):
        return self.values[0]

    def mean(self):
        return self.values.mean()

    def var(self):
        return self.values.var()

    def std(self):
        return math.sqrt(self.values.var())

    def cdf(self, x):
        if not hasattr(x, "__iter__"):
            return np.searchsorted(self.values, x) / len(self.values)
        else:
            return np.vectorize(lambda s: np.searchsorted(self.values, s) / len(self.values))(x)

    def ppf(self, q):
        if not hasattr(q, "__iter__"):
            return self.values[int(q * (len(self.values) - 1))]
        else:
            return self.values[np.vectorize(int)(q * (len(self.values) - 1))]
