import numpy as np
from scipy.stats import norm
from statsmodels.distributions.copula.elliptical import GaussianCopula


# Please refer to the appendix of the paper for the derivation of the formulas.


# s = np.arange(0, 1.2, .01)
# plt.plot(s, lnfcgaus(n, c, d, s))
def most_likly_sync(u):
    n = len(u)
    c = sum(norm.ppf(max(ui, 1e-3)) ** 2 for ui in u)
    d = sum(norm.ppf(max(ui, 1e-3)) for ui in u) ** 2
    roots = kubicroots(n, c, d)
    candidates = roots[roots.imag ** 2 < 1e-8].real
    assert len(candidates) > 0
    candidates = candidates[
        np.all([-1e-5 <= candidates, candidates <= 1 + 1e-5], axis=0)]  # only consider extrema in range [0,1]
    candidates = np.append(candidates, [0, 0.999])  # consider maxima at border
    s = candidates[lnfcgaus(n, c, d, candidates).argmax()]
    return max(min(s, 1), 0)  # limit to [0,1]


def most_likely_tau(u):
    return GaussianCopula.tau(None, np.array(most_likly_sync(u)))


def lnfcgaus(n, C, D, s):
    return -2 * (np.log(((1 + s * (n - 1)) * (1 - s) ** (n - 1)))
                 + C * ((1 + (n - 2) * s) / (1 - s) / (1 + (n - 1) * s) - 1)
                 - s / (1 - s) / (1 + (n - 1) * s) * (D - C))


def kubicroots(n, C, D):
    return np.roots([
        n * (n - 1) ** 2,  # s^3
        -n * (n - 1) * (n - 2) + (n - 1) ** 2 * C - (n - 1) * D,  # s^2
        -n * (n - 1) + 2 * (n - 1) * C,
        C - D
    ])

# https://www.sympygamma.com/input/?i=solve%28n*%28n-1%29**2*s**3%2B%28-n*%28n-1%29*%28n-2%29%2B%28n-1%29**2*C-%28n-1%29*D%29*s**2%2B%28-n*%28n-1%29%2B2*%28n-1%29*C%29*s%2BC-D%2Cs%29
# https://www.sympygamma.com/input/?i=solve%28simplify%28%281-s%29**2*%281%2Bs*%28n-1%29%29**2*diff%28log%28%281%2Bs*%28n-1%29%29*%281-s%29**%28n-1%29%29%2BC*%28%281%2B%28n-2%29*s%29%2F%281-s%29%2F%281%2B%28n-1%29*s%29-1%29%2B-s%2F%281-s%29%2F%281%2B%28n-1%29*s%29*%28D-C%29%2C+s%29%29%2Cs%29
