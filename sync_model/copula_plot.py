import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from statsmodels.distributions.copula.api import CopulaDistribution, GumbelCopula, GaussianCopula, IndependenceCopula


def plot_copula_3d(c):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    X = np.linspace(0, 1, 20 + 1)
    Y = np.linspace(0, 1, 20 + 1)
    X, Y = np.meshgrid(X, Y)
    Z = X * 0
    for i, x in enumerate(X):
        for j, y in enumerate(x):
            Z[i, j] = c.pdf((X[i, j], Y[i, j]))

    surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0, antialiased=True)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 2.5)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('C')
    # ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)


# c=IndependenceCopula()
c = GaussianCopula(0.5)
c = GaussianCopula(0.3)
# c = GumbelCopula(theta=1.0000000000001)
# c = GumbelCopula(theta=2)
c.plot_pdf()
plot_copula_3d(c)

plot_copula_samp(c)
