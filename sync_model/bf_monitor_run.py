import matplotlib.pyplot as plt
import numpy as np

import distribution as dist
from syncmodel import SyncModel
from bayes_filter_syncmodel import BayesFilterSM, System, plot_filters, plot_prob_exceed
import sync_groundtruth_generator as sgg

# s = SyncModel(100, 0.01, sample_size=5e6)
s = SyncModel(100, 0.01, sample_size=2e7)

t = np.arange(0, 300, 1)  # minutes
# r = sgg.constant_segments(300, [0.1, 0.2, 0.3])
r = sgg.constant_segments(300, [0, 0.2, 0.4])
# r = sgg.constant(300, 0.2)
# r = sgg.integrated_gaussian_noise(300, 0.02)  # "Intg. Gauss Noise (0.02)"

y = s.calc_samples_r(r, seed=0)
s.plot_totalpower_time(t, y, colorbar=True)

ylimits = [4]
# ylimits = [4, 8]
exceed_prob_gt = dict((limit, [s.calc_prob_exceed(r[i], limit, True) for i in range(len(r))]) for limit in ylimits)

prior = None  # dist.beta(1, 5)
bfs = []
bfs.append(BayesFilterSM(s, System.STATIC, name="Bayes Filter ($\\sigma=0$)", firstprior=prior))
# bfs.append(BayesFilterSM(s, System.NORMAL(len(s.rs), 0.005), name="Bayes Filter (0.005)", firstprior=prior))
# bfs.append(BayesFilterSM(s, System.NORMAL(len(s.rs), 0.01), name="Bayes Filter (0.01)", firstprior=prior))
bfs.append(BayesFilterSM(s, System.NORMAL(len(s.rs), 0.02), name="Bayes Filter ($\\sigma=0.02$)", firstprior=prior))
bfs.append(BayesFilterSM(s, System.NORMAL(len(s.rs), 0.05), name="Bayes Filter ($\\sigma=0.05$)", firstprior=prior))
# bfs.append(BayesFilterSM(s, System.NORMAL_PLUS_BASE(len(s.rs),0.02,.01), name="Bayes Filter (0.02+base1%)", firstprior=prior))
bfs.append(BayesFilterSM(s, System.NORMAL_PLUS_BASE(len(s.rs), 0.01, .01), name="Bayes Filter (0.01+base1%)",
                         firstprior=prior))
bfs.append(BayesFilterSM(s, System.NORMAL_PLUS_BASE(len(s.rs), 0.01, .01), name="Bayes Filter (0.01+base1%) - Argmax",
                         firstprior=prior, estimate_argmax=True))

for bf in bfs:
    bf.track_limit_probs(ylimits, normalized=True)
    bf.all_meas(y)

plot_filters(t, y, r, bfs)
plot_filters(t, y, r, bfs, plot_y=True, syncm=s)
# plot_filters(t,y,r,bfs,plot_y=True,plot_fill_quantiles=False)
# plot_filters(t,y,r,bfs,plot_y=True,gt_label="Ground Truth: Intg. Gauss Noise (0.02)")
plot_prob_exceed(t, bfs[0], exceed_prob_gt, ylimits)
plot_filters(t, y, r, bfs, plot_y=True, syncm=s, plot_p=True, exceed_prob_gt=exceed_prob_gt)
plot_filters(t, y, r, bfs, plot_y=True, syncm=s, plot_p=True, exceed_prob_gt=exceed_prob_gt, plot_fill_quantiles=False)
plot_filters(t, y, r, bfs[1], plot_y=True, syncm=s, plot_p=True, exceed_prob_gt=exceed_prob_gt)

plt.gcf().axes[0].set_ylim(0, 0.6)
plt.gcf().axes[2].set_ylim(1e-8)

# r_est = s.get_most_likely_r_at_y(y)
# r_est_beta15 = s.get_most_likely_r_at_y(y, dist.beta(1,5,plot=False))
# r_est_beta2 = s.get_most_likely_r_at_y(y,dist.beta(1.4,10))
# if not bf0:
#     ax1.plot(t,r_est,label="Bay. Inference - Uniform")
#     ax1.plot(t,r_est_beta15, label="Bay. Inference - Beta(1,8)",color="g")
# ax1.plot(t,r_est_beta2, label="Estimated; Beta(1.4,10)")
