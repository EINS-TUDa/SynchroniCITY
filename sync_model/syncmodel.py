import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pickle
import os
from hashlib import md5
from scipy import stats
from scipy.optimize import curve_fit
from statsmodels.distributions.copula.api import CopulaDistribution, GaussianCopula, ClaytonCopula, GumbelCopula, \
    FrankCopula

import graph2correlation as graph2corr
import distribution as dist
from ecdf import ECDF
import totalpower_cmap

datapath = os.path.dirname(__file__) + "/cache_sumdist_data/"


class SyncModel:

    def __init__(self, prosumers, r_res=.1, sample_size=1e5, bin_y=None, copula_type="GAUSS", copula_limit=None,
                 colormap="cool", corr_mat_type="ALL", corr_params=None, recalculate=False, savefile_fs=True,
                 savefile_ps=False):
        self.sample_size = int(sample_size)
        self.copula_type = copula_type
        self.copula_limit = copula_limit
        assert copula_limit is None or (len(copula_limit) == 2 and
                                        0 < copula_limit[0] < copula_limit[1] < 1)
        self.r_res = r_res
        if copula_type == "GAUSS":
            self.rs = np.linspace(0, 1, 1 + int(1 / r_res))
        else:
            self.rs = np.arange(r_res, 1, r_res)
        self.cmap_r = plt.colormaps.get(colormap)
        self.corr_mat_type = corr_mat_type
        self.corr_params = corr_params or {
            "sw_kn": 0.05,  # k mean degree, normalized to n
            "sw_p": 0.1,  # p rewire probability of SW net
        }
        self.default_prosumer = {
            "pdftype": "LOGNORMAL",
            "mean": 0.5,  # kW
        }
        if type(prosumers) is int:
            self.n_prosum = prosumers
            self.pdftype = self.default_prosumer["pdftype"]
            self.ecars = 0
            self.marginals = self.init_marginals(None)
        elif type(prosumers) is dict:
            self.n_prosum = prosumers["n"]
            self.pdftype = prosumers["pdftype"]
            self.default_prosumer["pdftype"] = prosumers["pdftype"]
            if "mean" in prosumers:
                self.default_prosumer["mean"] = prosumers["mean"]
            if "std" in prosumers:
                self.default_prosumer["std"] = prosumers["std"]
            self.ecars = prosumers.get("ecars", 0)  # 0-1 relative from n
            self.marginals = self.init_marginals(None)
        elif type(prosumers) is list:
            self.n_prosum = len(prosumers)
            self.pdftype = "LOGNORMAL" if all(prosumer["pdftype"] == "LOGNORMAL" for prosumer in prosumers) else None
            self.hash = self.prosumer_hash(prosumers)
            self.ecars = sum(prosumer.get("ecar", False) for prosumer in prosumers) / self.n_prosum
            self.marginals = self.init_marginals(prosumers)
        self.mean = sum(marginal.mean() for marginal in self.marginals)
        self.min = sum(marginal.ppf(0) for marginal in self.marginals)
        self.max = sum(marginal.ppf(1) for marginal in self.marginals)
        filename_ps = datapath + self.uniq_str() + "_ps"
        filename_fs = datapath + self.uniq_str() + "_fs"
        if os.path.isfile(filename_fs) and recalculate is False:
            print(f"Loading data from: {filename_fs}")
            self.fsy, self.fs = pickle.load(open(filename_fs, "rb"))
            self.bin_y = self.fsy[1] - self.fsy[0]
            if len(self.fs) < len(self.rs):
                self.rs = np.arange(0, 1, r_res)  # old without 1
            if os.path.isfile(filename_ps):
                self.ps = pickle.load(open(filename_ps, "rb"))
        else:
            if os.path.isfile(filename_ps) and recalculate is False:
                print(f"Loading data from: {filename_ps}")
                self.ps = pickle.load(open(filename_ps, "rb"))
            else:
                print("Start calculating Samples...")
                self.ps = self.calc_samples_all()
                if savefile_ps:
                    print(f"Saving to file: {filename_ps}")
                    pickle.dump(self.ps, open(filename_ps, "wb"))
            self.bin_y = bin_y or self.mean / 250
            self.fsy, self.fs = self.calc_hist_values(self.bin_y, self.ps)
            if savefile_fs:
                print(f"Saving to file: {filename_fs}")
                pickle.dump((self.fsy, self.fs), open(filename_fs, "wb"))
        print(f"Calc approx functions...")
        self.ds, parameters = self.hist_fit_to_func()
        print("Initialized " + self.__str__())
        print(f"Number Prosumer: {self.n_prosum}")
        print(f"Electric Vehicles: {self.ecars} ({self.n_prosum * self.ecars})")
        print(f"Mean: {self.mean} kW")

    def __str__(self):
        return f"SyncModel '{self.uniq_str()}'"

    def uniq_str(self):
        return "_".join(filter(None, [
            f"n{self.n_prosum:d}",
            f"{self.pdftype if self.pdftype else self.hash}",
            f"e{self.ecars:g}",
            f"{self.copula_type}" if self.copula_type != "GAUSS" else None,
            f"{self.corrmat_str()}",
            "-".join(f"{x:g}".lstrip("0") for x in self.copula_limit) if self.copula_limit else None,
            f"r{self.r_res:g}",
            f"s{self.sample_size:.1E}",
        ]))

    def corrmat_str(self):
        prms = None
        if self.corr_mat_type == "SMALLWORLD":
            prms = f'kn{self.corr_params["sw_kn"]:g},p{self.corr_params["sw_p"]:g}'
        elif self.corr_mat_type == "SCALEFREE":
            prms = f'm{self.corr_params["sf_m"]:g}'
        return f'{self.corr_mat_type}{f"({prms})" if prms else ""}'

    def prosumer_hash(self, prosumers):
        return md5("*".join(repr(prosumer.items()) for prosumer in prosumers).encode('utf-8')).hexdigest()

    def get_dist(self, prosumer):
        if prosumer["pdftype"] == "LOGNORMAL":
            marg = dist.lognormal_EV(prosumer["mean"], prosumer.get("std", 1.5 * prosumer["mean"]) ** 2)
        elif prosumer["pdftype"] == "BETA":
            marg = dist.beta_EV(prosumer["mean"], prosumer.get("std", 1.5 * prosumer["mean"]) ** 2, scale=15)
        elif prosumer["pdftype"] == "ECDF":
            marg = ECDF(prosumer["data"])
        else:
            raise Exception("pdftype: " + self.pdftype)
        if self.copula_limit:
            orig_ppf = marg.ppf
            marg.ppf = lambda x: orig_ppf(self.copula_limit[0] + x * (self.copula_limit[1] - self.copula_limit[0]))
        return marg

    def init_marginals(self, prosumers):
        if prosumers:
            return [
                *[self.get_dist(prosumer) for prosumer in prosumers],
                *[dist.e_car(load=11, charging_hrs=1) for prosumer in prosumers if prosumer.get("ecar")],
            ]
        else:
            return [
                *[self.get_dist(self.default_prosumer) for _ in range(self.n_prosum)],
                *[dist.e_car(load=11, charging_hrs=1) for _ in range(int(self.n_prosum * self.ecars))],
            ]

    def create_copula(self, r, corr_mat_type=None, corr_params=None):
        n = len(self.marginals)
        corr_mat_type = corr_mat_type or self.corr_mat_type
        corr_params = corr_params or self.corr_params
        if corr_mat_type == "ALL":
            corr_mat = np.eye(n) + r * (1 - np.eye(n))
        elif corr_mat_type == "SMALLWORLD":
            corr_mat = graph2corr.calc_corr_mat(
                graph2corr.get_strog_graph(n, int(corr_params["sw_kn"] * n), corr_params["sw_p"]), r)
        elif corr_mat_type == "SCALEFREE":
            corr_mat = graph2corr.calc_corr_mat(graph2corr.get_barabasi_graph(n, corr_params["sf_m"]), r)
        else:
            raise Exception(corr_mat_type)
        if self.copula_type == "GAUSS":
            copula = GaussianCopula(corr=corr_mat, k_dim=n, allow_singular=True)
        elif self.copula_type == "CLAYTON":
            copula = ClaytonCopula(ClaytonCopula.theta_from_tau(None, r), k_dim=n)
        elif self.copula_type == "GUMBEL":
            copula = GumbelCopula(GumbelCopula.theta_from_tau(None, r), k_dim=n)
        elif self.copula_type == "FRANK":
            copula = FrankCopula(FrankCopula.theta_from_tau(None, r), k_dim=n)
        else:
            raise Exception(self.copula_type)
        return CopulaDistribution(copula=copula, marginals=self.marginals)

    def calc_samples_all(self, rs=None):
        psums = []
        for r in (rs or self.rs):
            print(f"r={r:g}")
            c = self.create_copula(r)
            BUFSIZE = 1e7
            if c.k_vars * self.sample_size > BUFSIZE:
                psum = []
                for _ in range(round(c.k_vars * self.sample_size / BUFSIZE)):
                    sample = c.rvs(round(BUFSIZE / c.k_vars))
                    psum = np.concatenate([psum, sample.sum(1)])
            else:
                sample = c.rvs(self.sample_size)
                psum = sample.sum(1)
            psums.append(psum)
        return psums

    def calc_samples_r(self, r_time, corr_mat=None, corr_params=None, seed=0):
        np.random.seed(seed)
        psums = []
        for r in r_time:
            c = self.create_copula(r, corr_mat, corr_params)
            sample = c.rvs(2)[0]  # bug with rvs(1)
            psum = sample.sum()
            psums.append(psum)
        return psums

    def calc_hist_values(self, bin_kw, ps):
        bins = np.arange(0, ps[-1].max(), bin_kw)
        fs = []
        for i, r in enumerate(self.rs):
            fs.append(np.diff(np.searchsorted(np.sort(ps[i]), bins)) / len(ps[i]) / bin_kw)
        return bins[0:-1], fs

    def hist_fit_to_func(self, plot=False):
        if self.pdftype == "LOGNORMAL":
            func = lambda x, c, d: stats.lognorm.pdf(x, c, 0, d)  # fix loc to 0
            p0 = [1, self.mean]
        else:
            return None, None
        ds = []
        parameters = []
        for i, r in enumerate(self.rs):
            xdata, ydata = self.fsy, self.fs[i]
            popt, pcov = curve_fit(func, xdata, ydata, p0=p0)
            if pcov.__abs__().sum() > 1e-3:
                print("WARNING: pcov > 1e-3")
            if plot:
                print(f"r={r:g}  Parameters: {popt} ; pcov = {pcov.__abs__().sum()}")
                plt.plot(xdata, ydata, 'b-', label='data')
                plt.plot(xdata, func(xdata, *popt), 'r--', label="func")
                plt.legend()
            parameters.append(popt)
            ds.append(lambda x, params=popt: func(x, *params))
        if plot:
            plt.ylim(0)
            plt.xlim(0, 4 * self.mean)
            plt.show()
        return ds, parameters

    def get_fsy_index(self, y):
        return np.minimum(np.searchsorted(self.fsy, y), len(self.fsy) - 1)

    def get_likelihood_at_y(self, y, normalized=False):
        if normalized:
            y = y * self.mean
        if self.ds:
            return np.array([self.ds[i](y) for i, r in enumerate(self.rs)])
        else:
            return np.array([self.fs[i][self.get_fsy_index(y)] for i, r in enumerate(self.rs)])

    def get_likelihood_at_y_prior(self, y, prior, normalized=False):
        return self.get_likelihood_at_y(y, normalized) * prior.pdf(self.rs).reshape(
            [len(self.rs), 1])  # reshape for y=array

    def get_most_likely_r_at_y(self, y, prior=None, normalized=False):
        likelihood = self.get_likelihood_at_y_prior(y, prior, normalized) \
            if prior else self.get_likelihood_at_y(y, normalized)
        return np.array(self.rs[likelihood.argmax(0)])

    def get_quantile_r_at_y(self, y, q, prior=None, normalized=False):
        likelihood = self.get_likelihood_at_y_prior(y, prior, normalized) \
            if prior else self.get_likelihood_at_y(y, normalized)

        def getquantile(dist, q):
            c = dist.cumsum()
            return max(c.searchsorted(q * c[-1]) - 1, 0)  # should work without max in future

        if not hasattr(y, "__iter__"):
            idxs = getquantile(likelihood, q)
        else:  # y=list
            idxs = [getquantile(l, q) for l in likelihood.transpose()]  # handle idx=len(rs)
        return self.rs[idxs]

    def get_y_for_r(self, r):
        idx_min = np.searchsorted(self.fsy, self.min)
        idx_mean = np.searchsorted(self.fsy, self.mean)
        likely_r_left = self.get_most_likely_r_at_y(self.fsy)[idx_min + 1:idx_mean]  # skip y=0
        likely_r_right = self.get_most_likely_r_at_y(self.fsy)[idx_mean:len(self.fsy) // 2]
        if not np.all(np.diff(likely_r_left) <= 0):
            print("WARNING: likely_r_left not decreasing")
        if not np.all(np.diff(likely_r_right) >= 0):
            print("WARNING: likely_r_right not increasing")
        idxleft = np.where(likely_r_left < r)[0]
        idxright = np.where(likely_r_right < r)[0]
        if len(idxleft) == 0 or len(idxright) == 0:
            print(f"WARNING: get_y_for_r not found r={r:g}")
            return None
        else:
            return (self.fsy[idx_min + 1 + idxleft.min()], self.fsy[idx_mean + idxright.max()])

    def calc_pry(self, prior, y):
        pry = self.get_likelihood_at_y(y) * prior.pdf(self.rs)
        return pry / pry.sum() / self.r_res  # normalize

    def calc_prob_exceed(self, r, ylimit, normalized=False):
        ylimit = ylimit * self.mean if normalized else ylimit
        if r is None:  # all
            return [self.fs[i][self.get_fsy_index(ylimit):].sum() * self.bin_y for i, r in enumerate(self.rs)]
        else:
            ri = int(r / self.r_res)
            assert 0 <= ri < len(self.rs)
            return self.fs[ri][self.get_fsy_index(ylimit):].sum() * self.bin_y

    def calc_prob_exceed_restimate(self, estimate, ylimit, normalized=False):
        estimate /= sum(estimate)
        probs = self.calc_prob_exceed(None, ylimit, normalized)
        assert len(estimate) == len(probs)
        return sum(estimate * probs)

    def calc_py(self, prior, normalized=False):
        return np.array(
            [self.get_likelihood_at_y_prior(y, prior, normalized=False).sum() for y in self.fsy]) * self.r_res

    def plot_py(self, prior, normalized=True, log=False):
        fig, ax = plt.subplots()
        ax.plot(self.fsy, self.calc_py(prior, normalized))
        ax.set_xlim(0, self.mean * 4)
        if log:
            ax.set_yscale('log')

    def plot_pry(self, prior, yss, normalized=False, cmap="turbo"):
        mean = 1 if normalized else self.mean
        fig, ax = plt.subplots(1, 1)
        # ax.set_prop_cycle(color=plt.colormaps.get(cmap)(np.linspace(0.1, 0.9, len(yss))))
        for y in (yss if type(yss) is list else [yss]):
            ax.plot(self.rs, self.calc_pry(prior, y * self.mean if normalized else y),
                    linestyle=y == mean and "solid" or y < mean and "dashed" or y > mean and "dotted",
                    label=f"y={y}",
                    color=totalpower_cmap.get(y if normalized else y / mean))
        plt.legend()
        plt.xlabel("Synchronization")
        plt.ylabel("p(s|y)")
        plt.xlim(0, 1)
        plt.ylim(0)
        plt.show()

    def plot_densities_and_most_likely_r(self, quantiles=[0.1, 0.9], prior=None, **kwargs):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        self.plot_total_densities(ax=ax1, **kwargs)
        ax2.plot(self.fsy, self.get_quantile_r_at_y(self.fsy, .5, prior), label="median")
        ax2.plot(self.fsy, self.get_most_likely_r_at_y(self.fsy, prior), label="argmax")
        if quantiles:
            ax2.fill_between(self.fsy,
                             self.get_quantile_r_at_y(self.fsy, quantiles[0], prior),
                             self.get_quantile_r_at_y(self.fsy, quantiles[1], prior),
                             label=f"Q:{quantiles[0]:g}-{quantiles[1]:g}",
                             alpha=0.2)
            ax2.legend()
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Most likely s")
        ax2.set_xlabel(ax1.get_xlabel())
        ax1.set_xlabel("")
        plt.show()
        return ax1, ax2

    def plot_total_densities(self, mcshist=False, normalized=False, rs_fac=None, ax=None, rs=None, log=False,
                             colorbar=False, ymax=None, **kwargs):
        if mcshist is False and self.ds is None:
            mcshist = True
            print("No Approx function - Use Histogram")
        if ax is None:
            if colorbar:
                fig, (ax, ax_colorbar) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [10, 1]})
            else:
                fig, ax = plt.subplots()
        for i, r in enumerate(self.rs):
            if rs is not None and r not in rs:
                continue
            ax.plot(self.fsy / self.mean if normalized else self.fsy,
                    (self.fs[i] if mcshist else self.ds[i](self.fsy)) * (self.mean if normalized else 1),
                    color=self.getcolor_r(r),
                    label=f"s={r:g}" if (rs_fac is None) or (i % rs_fac == 0) else None,
                    )
        for i, r in enumerate(self.rs):
            if rs_fac and i % rs_fac == 0:
                ax.plot(self.fsy / self.mean if normalized else self.fsy,
                        (self.fs[i] if mcshist else self.ds[i](self.fsy)) * (self.mean if normalized else 1),
                        color="k", linewidth=0.4)
        if colorbar:
            cmap, norm = self.create_cmap_y(normalized, ymax)
            matplotlib.colorbar.ColorbarBase(ax_colorbar, cmap=cmap, norm=norm, orientation="horizontal")
            # plt.imshow(s.fsx*np.ones([2000,1]),cmap)
        ax.legend()
        if ymax is None:
            ymax = 4 if normalized else self.mean * 4
        ax.set_xlim(0, ymax)
        ax.set_ylim(0)
        ax.set_xlabel("Total Power [normalized to mean]" if normalized else "Total Power (kW)")
        ax.set_ylabel("Probability Density")
        if log:
            ax.set_yscale('log')
            plt.grid()
        plt.show()
        return ax

    def plot_prob_exceed(self, mcshist=False, ylimits=[2, 4, 8, 16], normalized=True, log=True, ):
        if mcshist is False and self.ds is None:
            mcshist = True
            print("No Approx function - Use Histogram")
        fig, ax = plt.subplots()
        for y in ylimits:
            pover = self.calc_prob_exceed(None, y, normalized)
            plt.plot(self.rs, pover, label=f"P(Sum > {f'{y:g} Mean' if normalized else f'{y:.0f} kW'})")
        plt.xlabel("Synchronization")
        plt.xlim(0, 1)
        plt.legend()
        plt.grid()
        if log:
            ax.set_yscale('log')

    def plot_densities_and_pexceed(self, mcshist=False, ylimits=[2, 4, 8, 16], normalized=True, rs_fac=None, log=False):
        ax = self.plot_total_densities(mcshist=mcshist, normalized=normalized, rs_fac=rs_fac, log=log)
        ax.set_xlim(0, ylimits[-1] * 1.1)
        prop_cycle = plt.rcParams['axes.prop_cycle']
        plt_colors = prop_cycle.by_key()['color']
        for i, y in enumerate(ylimits):
            ax.vlines(y, 0, self.fs[0].max() * (self.mean if normalized else 1), linestyle="--", color=plt_colors[i])
        self.plot_prob_exceed(mcshist, ylimits, normalized)

    def plot_sync_total_2d_density(self, normalized=False, log=True, ymax=None):
        fig, ax = plt.subplots()
        pry = pd.DataFrame(np.asarray(self.fs).T, index=self.fsy / (self.mean if normalized else 1), columns=self.rs)
        ax.imshow(pry, origin="lower", aspect="auto", norm=LogNorm(clip=True) if log else None,
                  extent=[0, 1, self.fsy[0] / (self.mean if normalized else 1),
                          self.fsy[-1] / (self.mean if normalized else 1)])
        if ymax is None:
            ymax = 4 if normalized else self.mean * 4
        plt.ylim(None, ymax)
        plt.xlabel("Synchronization")
        plt.ylabel("Total Power (kW)")

    def plot_totalpower_time(self, t, y, rsmax=1, ax=None, colorbar=True, ax_colorbar=None, normalized=False,
                             adjtimex=False):
        if ax is None:
            if colorbar:
                fig, (ax, ax_colorbar) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [10, 1]})
            else:
                fig, ax = plt.subplots()
        ax.hlines(self.mean, t[0], t[-1], color="k", linestyle="--")
        if rsmax:
            rs = np.arange(0.1, rsmax, .1)
            for ri in rs:
                ys = self.get_y_for_r(ri)
                if ys:
                    ax.hlines(ys[0], t[0], t[-1], color=self.getcolor_r(ri), linestyle="--", label=f"s={ri:g}")
                    ax.hlines(ys[1], t[0], t[-1], color=self.getcolor_r(ri), linestyle="--")
            ax.legend()
        ax.plot(t, y, color="tab:orange")
        ax.set_ylabel("Total Power (kW)")
        ax.set_xlabel("Time")
        ax.set_ylim(0)
        ax.set_xlim(t[0])
        ax.grid()
        if adjtimex:
            ticks = t[np.linspace(0, len(t) - 1, 11).astype("int")]
            ax.set_xticks(ticks, [
                tick.strftime("%H:%M\n%Y-%m-%d") if tick == t[0] else tick.strftime("%H:%M")
                # else "" if tick == ticks[1]
                for tick in ticks])
        if colorbar and ax_colorbar:
            cmap, norm = self.create_cmap_y(normalized, self.mean * 4)
            matplotlib.colorbar.ColorbarBase(ax_colorbar, cmap=cmap, norm=norm, orientation="vertical")

    def plot_totalpower_time_and_most_likely_r(self, t, y, quantiles=[0.1, 0.9], prior=None, **kwargs):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        kwargs.setdefault("rsmax", 0.6)
        self.plot_totalpower_time(t, y, ax=ax1, **kwargs)
        ax2.plot(t, self.get_most_likely_r_at_y(y))
        if quantiles:
            ax2.fill_between(t,
                             self.get_quantile_r_at_y(y, quantiles[0], prior),
                             self.get_quantile_r_at_y(y, quantiles[1], prior),
                             label=f"Q:{quantiles[0]:g}-{quantiles[1]:g}",
                             alpha=0.2)
            ax2.legend()
        ax2.set_ylim(0)
        ax2.set_ylabel("Most likely s")
        ax2.grid()
        ax2.set_xlabel("Time")
        ax1.set_xlabel("")

    def getcolor_r(self, r):
        getcolcmap = lambda s: self.cmap_r(int((self.cmap_r.N - 1) * s))
        return getcolcmap((r - self.rs[0]) / (self.rs[-1] - self.rs[0]))

    def create_cmap_y(self, normalized=False, ymax=None):
        return totalpower_cmap.create_cmap(
            self.fsy[self.fsy < (self.mean * ymax if normalized else ymax or 3 * self.mean)] / (
                self.mean if normalized else 1),
            1 if normalized else self.mean
        )
