import matplotlib.pyplot as plt
import pandas as pd
import pickle

from readdata import readdata
from copula_unitvars import calc_unitvars
from copula_syncest_gaus import most_likly_sync as mls_gaus, most_likely_tau as mlt_gaus
from copula_syncest_arch import most_likely_tau as mlt_arch

sync_data_path = "sync_results/"

year = 2019
data, psum = readdata("1min", year, with_hp=False)
# data, psum = readdata("10s",with_hp=False,dateend=24 * 60 * 60)

data_ue, data_ue8, data_ul, ldists, data_uln = calc_unitvars(data)

sync_gaus_ln = pd.Series(index=data.index, dtype="float64")
sync_gaus_emp = pd.Series(index=data.index, dtype="float64")
sync_gaus_cl8 = pd.Series(index=data.index, dtype="float64")
sync_gum_ln = pd.Series(index=data.index, dtype="float64")
sync_gum_emp = pd.Series(index=data.index, dtype="float64")
sync_gum_cl8 = pd.Series(index=data.index, dtype="float64")
sync_cly_ln = pd.Series(index=data.index, dtype="float64")
sync_cly_emp = pd.Series(index=data.index, dtype="float64")
sync_cly_cl8 = pd.Series(index=data.index, dtype="float64")
sync_frk_ln = pd.Series(index=data.index, dtype="float64")
sync_frk_emp = pd.Series(index=data.index, dtype="float64")
sync_frk_cl8 = pd.Series(index=data.index, dtype="float64")

dates = data.index
# dates = data.index[:len(data)//24]
# dates = data.index[:100]
for date in dates:
    # sync_gaus_ln[date] = mlt_gaus(data_ul.loc[date])
    sync_gaus_emp[date] = mlt_gaus(data_ue.loc[date])
    # sync_gaus_cl8[date] = mlt_gaus(data_ue8.loc[date])
    # sync_gum_ln[date]=mlt_arch(data_ul.loc[date], "gumbel")
    sync_gum_emp[date] = mlt_arch(data_ue.loc[date], "gumbel")
    # sync_gum_cl8[date]=mlt_arch(data_ue8.loc[date], "gumbel")
    # sync_cly_ln[date]=mlt_arch(data_ul.loc[date], "clayton")
    sync_cly_emp[date] = mlt_arch(data_ue.loc[date], "clayton")
    # sync_cly_cl8[date] = mlt_arch(data_ue8.loc[date], "clayton")
    # sync_frk_ln[date]=mlt_arch(data_ul.loc[date], "frank")
    sync_frk_emp[date] = mlt_arch(data_ue.loc[date], "frank")
    # sync_frk_cl8[date]=mlt_arch(data_ue8.loc[date], "frank")

pickle.dump(sync_gaus_ln, open(f"{sync_data_path}sync_{year}_gaus_ln", "wb"))
pickle.dump(sync_gaus_cl8, open(f"{sync_data_path}sync_{year}_gaus_cl8", "wb"))
pickle.dump(sync_cly_cl8, open(f"{sync_data_path}sync_{year}_cly_cl8", "wb"))
