import numpy as np


def clip(x, lower, upper):
    if x < lower:
        return lower
    elif x > upper:
        return upper
    else:
        return x


def integrate_and_clip(x, start=0, bounds=(0, 1)):
    y = []
    s = start
    for xi in x:
        s = clip(s + xi, bounds[0], bounds[1])
        y.append(s)
    return y


def gaussian_noise(n, sigma, ):
    return sigma * np.random.randn(n)


def integrated_gaussian_noise(n, sigma, ):
    return integrate_and_clip(gaussian_noise(n, sigma))


def constant(n, val):
    return [val] * n


def constant_segments(n, values):
    return np.concatenate([[val] * int(n / len(values)) for val in values])
