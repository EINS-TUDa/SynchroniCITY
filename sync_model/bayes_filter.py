import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from filterpy import discrete_bayes as bf
from typing import List


# Enhanced Version of filterpy for a common Bayes Filter

class System:
    STATIC = [1]

    @staticmethod
    def STEP(length):
        return [1] * (length * 2 - 1)

    @staticmethod
    def NORMAL(length, sigma):
        x = np.linspace(-1, 1, 2 * length + 1)
        return stats.norm(loc=0, scale=sigma).pdf(x)  # sum=length

    @staticmethod
    def NORMAL_PLUS_BASE(length, sigma, base):
        x = np.linspace(-1, 1, 2 * length + 1)
        return stats.norm(loc=0, scale=sigma).pdf(x) + base
        # sum=length+(base*length) -> base is relative fraction (if small)


class BayesFilter:

    def __init__(self, state_values: List[float], likelihood, system=System.STATIC, firstprior=None, logging=True,
                 name=None, estimate_quantiles=[0.1, 0.9], estimate_argmax=False):
        self.name = name
        self.state_label = "State"
        self.state_values = state_values
        self.r_res = self.state_values[1] - self.state_values[0]
        self.likelihood = likelihood  # function(measurement)
        assert len(system) % 2 == 1, "Length of system should be uneven"
        assert np.array(system).argmax() == (len(system) - 1) / 2, "Maximum should be in the middle"
        self.kernel = bf.normalize(system)  # means system behavior (like prob dist)
        if firstprior is None:
            self.prior = bf.normalize(np.ones(len(self.state_values)))
        elif hasattr(firstprior, "pdf"):
            self.prior = firstprior.pdf(self.state_values)
        else:
            self.prior = firstprior
        self.posterior = None
        self.estimates_quantiles = dict((q, []) for q in estimate_quantiles)
        self.estimate_argmax = estimate_argmax
        self.logging = logging
        if logging:
            self.priors = []
            self.posteriors = []
            self.estimates = []

    def __str__(self):
        return self.name or "Bayes Filter"

    def new_meas(self, measurement):
        likelihood = self.likelihood(measurement)
        self.posterior = bf.update(likelihood, self.prior)
        self.prior = bf.normalize(
            bf.predict(self.posterior, offset=0, kernel=self.kernel, mode="constant"))  # convolve can change to sum!=1
        if self.logging:
            self.posteriors.append(self.posterior)
            self.priors.append(self.prior)
            self.estimates.append(
                self.get_estimate_argmax() if self.estimate_argmax else self.get_estimate_median())
            for q in self.estimates_quantiles:
                self.estimates_quantiles[q].append(self.get_estimate_quantile(q))

    def all_meas(self, y):
        for yi in y:
            self.new_meas(yi)

    def get_estimate_argmax(self):
        return self.state_values[self.prior.argmax()]

    def get_estimate_median(self):
        return self.get_estimate_quantile(0.5)

    def get_estimate_quantile(self, q):
        return self.state_values[self.prior.cumsum().searchsorted(q)]

    def get_estimate_dist(self, k=None):
        if k is None:
            dist = self.prior
        else:
            dist = self.priors[k]
        return dist / self.r_res

    def plot_estimate(self, k=None):
        f = plt.figure()
        plt.plot(self.state_values, self.get_estimate_dist(k), label=f"p(s|y[1:{k + 1 if type(k) is int else 'k'}])")
        plt.xlabel(self.state_label)
        plt.xlim(0, 1)
        plt.ylim(0)
        # plt.ylabel("Probability Density")
        plt.grid()
        plt.legend()
