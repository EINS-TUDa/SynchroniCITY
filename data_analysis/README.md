This folder contains the code for the electricity data and synchronization analysis
of [this dataset](https://www.nature.com/articles/s41597-022-01156-1).

The code is mostly adjusted for the usage with the python console.

The most important files are:

- metadata_analysis.py: general analysis of the dataset
- readdata.py: a function to return a pandas dataframe with the household data
- copula_unitvars.py: transformation of the data to unit variables
- copula_syncest_gaus.py: gaussian copula estimation
- copula_syncest_arch.py: archimedean copula estimation by numerical optimization
- run_copula_sync_est.py: script to run the synchronization estimation
- sync_analysis.py: analysis of the synchronization results for different configurations
- sync_plot.py: functions for plotting the synchronization results
- diversity_factor.py: analysis of the diversity factor
- monitor_run.py: application of bayes filter monitoring with the data
- ecar_scenario.py: scenario analysis of hypothetical EV home-charging

The synchronization estimation was calculated in 12 different configurations (copula and marginal type) for 2019 and in
3 configurations for 2020.
The results are stored in the folder `sync_results`.
