import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.correlation_tools as corr_tools


# G = nx.barabasi_albert_graph(1000,1)
def get_barabasi_graph(n, m):
    return nx.barabasi_albert_graph(n=n, m=m)


# G = nx.watts_strogatz_graph(n=100, k=4, p=0.05)
def get_strog_graph(n, k, p):
    return nx.watts_strogatz_graph(n=n, k=k, p=p)


def calc_corr_mat(G, base_corr):
    d = nx.shortest_path_length(G)  # TODO handle inf
    m = np.eye(len(G))
    for n1, dists in d:
        for n2, dist in dists.items():
            m[n1, n2] = base_corr ** dist
    mineig = np.linalg.eigvals(m).min()
    # print("Min eig value:", mineig)
    if mineig < 1e-15:
        # print("Calc nearest corrmat")
        m = get_nearst_corrmat_fast(m)
    return m


# https://stats.stackexchange.com/questions/295093/given-an-adjacency-matrix-how-can-we-fit-a-covariance-matrix-based-on-that-for
def get_corr_mat_by_precmat(G, base_corr):
    meandeg = sum([d[1] for d in G.degree]) / len(G)
    m = nx.adjacency_matrix(G).toarray()
    p = np.eye(len(m)) - (base_corr / meandeg) * m  # precision matrix
    cov = np.linalg.inv(p)  # numeric problems often even with base_corr<1
    for i, row in enumerate(cov):
        for j, val in enumerate(row):
            if i != j:
                cov[i, j] /= math.sqrt(cov[i, i] * cov[j, j])
    for i, row in enumerate(cov):
        for j, val in enumerate(row):
            if i == j:
                cov[i, j] = 1
    assert abs(np.trace(cov) - len(m)) < 1e2
    mineig = np.linalg.eigvals(cov).min()
    print("Min eig value:", mineig)
    assert mineig > 1e-15
    return cov


def get_adjmatrix(G):
    m = nx.adjacency_matrix(G).toarray()
    return m


def get_nearst_corrmat(m):
    return corr_tools.corr_nearest(m)


def get_nearst_corrmat_fast(m):
    return corr_tools.corr_clipped(m)


def plot_sw(n, k, p):
    G = get_strog_graph(n, k, p)
    pos = nx.circular_layout(G)
    plt.figure(figsize=(12, 12))
    nx.draw_networkx(G, pos)
    # nx.draw_networkx(G)


def plot_ba(n, m):
    G = get_barabasi_graph(n, m)
    nx.draw_networkx(G)


def test_sw():
    avgp = []
    clu = []
    for p in np.arange(0, 1, 0.01):
        G = nx.watts_strogatz_graph(n=1000, k=8, p=p)
        avgp.append(nx.average_shortest_path_length(G))
        clu.append(nx.average_clustering(G))

        plt.plot(avgp)
        plt.plot(clu)
